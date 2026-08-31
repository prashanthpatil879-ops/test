# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../../common/nb_load_rda_package

# COMMAND ----------

from rda.utils.logger import get_notebook_logger
logger = get_notebook_logger()
logger.info("HA Ineffectiveness - Partial DV01 curation loading started...")

# COMMAND ----------

dataset_key = "ha_ineffectiveness_curation"

# COMMAND ----------

dataset_cfg = env_config.get(f"curation.{dataset_key}")

# COMMAND ----------

import re
from pyspark.sql import functions as F, Window
from rda.io.unity_catalog import UnityCatalog
import rda.utils.spark_utils as su

uc = UnityCatalog(env_config, spark)

# COMMAND ----------

dv01_fi_df = uc.read_latest(dataset_cfg.inputs.dv01_fi)
dv01_derivatives_df = uc.read_latest(dataset_cfg.inputs.dv01_derivatives)
dv01_liab_post_hp_df = uc.read_latest(dataset_cfg.inputs.dv01_liab_post_hp)
fx_rates_df = uc.read_latest(dataset_cfg.inputs.fx_rates)
heirarchy_mapping_df = uc.read_latest(dataset_cfg.inputs.heirarchy_mapping_latest)

# COMMAND ----------

# HA DV01 mapping_NT_LLP.xlsx - hardcoded until the LLP mapping table is ingested.
# Replace this cell with a uc.read_latest(dataset_cfg.inputs.ha_dv01_llp) once it lands.
HA_DV01_LLP_MAPPING = [
    ("GH_14", 28, "USD", "HONG KONG SURPLUS"),
    ("MB_01", 58, "CAD", "ManuBank"),
    ("GH_01", 1, "CAD", "CA CAD-Guaranteed"),
    ("GH_09", 1, "CAD", "CA CAD-Guaranteed"),
    ("GH_01", 19, "USD", "JH USD-Guaranteed"),
    ("GH_09", 19, "USD", "JH USD-Guaranteed"),
    ("GH_01", 176, "AUD", "MLJ AUD-Guaranteed"),
    ("GH_01", 176, "JPY", "MLJ JPY-Guaranteed"),
    ("GH_09", 176, "JPY", "MLJ JPY-Guaranteed"),
    ("GH_01", 176, "USD", "MLJ USD-Guaranteed"),
    ("GH_01", 391, "AUD", "MLRL AUD-Guaranteed"),
    ("GH_01", 391, "JPY", "MLRL JPY-Guaranteed"),
    ("GH_09", 391, "JPY", "MLRL JPY-Guaranteed"),
    ("GH_01", 391, "USD", "MLRL USD-Guaranteed"),
    ("GH_02", 248, "USD", "CA CAD-Guaranteed"),
    ("GH_02", 176, "AUD", "MLJ JPY-Guaranteed"),
    ("GH_02", 176, "CAD", "MLJ JPY-Guaranteed"),
    ("GH_02", 176, "EUR", "MLJ JPY-Guaranteed"),
    ("GH_02", 176, "GBP", "MLJ JPY-Guaranteed"),
    ("GH_02", 176, "NOK", "MLJ JPY-Guaranteed"),
    ("GH_02", 176, "USD", "MLJ JPY-Guaranteed"),
    ("GH_02", 391, "AUD", "MLRL JPY-Guaranteed"),
    ("GH_02", 391, "CHF", "MLRL JPY-Guaranteed"),
    ("GH_02", 391, "EUR", "MLRL JPY-Guaranteed"),
    ("GH_02", 391, "GBP", "MLRL JPY-Guaranteed"),
    ("GH_02", 391, "NOK", "MLRL JPY-Guaranteed"),
    ("GH_02", 391, "USD", "MLRL JPY-Guaranteed"),
    ("GH_02", 391, "SEK", "MLRL JPY-Guaranteed"),
]

llp_mapping_df = spark.createDataFrame(
    HA_DV01_LLP_MAPPING,
    "PROGRAM_CODE string, LEGAL_ID int, SECURITY_CURRENCY string, LOWEST_LEVEL_PORTFOLIO_NAME string"
).distinct()

display(llp_mapping_df)

# COMMAND ----------

def unpivot(df):
    tenors = [c for c in df.columns if re.fullmatch(r"\d+M", c)]

    if not tenors:
        raise ValueError(f"No tenor columns found in: {df.columns}")

    logger.info(f"Unpivoting {len(tenors)} tenor columns: {tenors}")
    expr = "stack({0}, {1}) as (TENOR_CD, DV01_VALUE)".format(
        len(tenors), ", ".join([f"'{c}', CAST(`{c}` AS DOUBLE)" for c in tenors])
    )
    return df.select("ASSET_TYPE", "INCEPTION_DATE", "PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY", "SOURCE", F.expr(expr))


def validate_lookup(df, lookup_col, key_cols, source):
    unmatched = df.filter(F.col(lookup_col).isNull()).select(*key_cols).distinct().collect()

    if unmatched:
        raise ValueError(f"{source} lookup failed for {len(unmatched)} key(s): {[r.asDict() for r in unmatched]}")

    logger.info(f"{source} lookup matched every row")

# COMMAND ----------

dv01_unpivot_df = (
    unpivot(dv01_fi_df)
    .unionByName(unpivot(dv01_derivatives_df))
    .unionByName(unpivot(dv01_liab_post_hp_df))
    .withColumn("DV01_VALUE", F.coalesce(F.col("DV01_VALUE"), F.lit(0.0)))
    .withColumn("INCEPTION_DATE", F.to_date(F.col("INCEPTION_DATE")))
    .withColumn("PROGRAM_CODE", F.upper(F.trim(F.col("PROGRAM_CODE"))))
    .withColumn("SECURITY_CURRENCY", F.upper(F.trim(F.col("SECURITY_CURRENCY"))))
    .withColumn("LEGAL_ID", F.col("LEGAL_ID").cast("int"))
)
display(dv01_unpivot_df)

# COMMAND ----------

# TOTAL_DV01 = sum of DV01 across the three ALM datasets, per tenor, grouped on
# PROGRAM_CODE + LEGAL_ID + SECURITY_CURRENCY. The datasets are disjoint on
# PROGRAM_CODE (GH_01 only in liab_post_hp + derivatives, the rest only in
# fi + derivatives), so this reproduces the per-combination source rules in the
# "Calculation sum_partial_dv01" column of HA DV01 mapping_NT_LLP.xlsx.
total_dv01_df = (
    dv01_unpivot_df
    .withColumn("REPORT_DATE", F.last_day(F.col("INCEPTION_DATE")))
    .groupBy("REPORT_DATE", "PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY", "TENOR_CD")
    .agg( F.sum("DV01_VALUE").alias("TOTAL_DV01") )
    .withColumn("TERM_YEARS", F.regexp_extract(F.col("TENOR_CD"), r"^(\d+)M$", 1).cast("int") / F.lit(12))
)

display(total_dv01_df)

# COMMAND ----------

# LLP from the hardcoded mapping, then TAX_RATE from the hierarchy mapping keyed
# on that LLP. su.normalize_text absorbs the case difference between the mapping
# ("ManuBank") and the hierarchy table ("Manubank"), same as non_ha.py does.
tax_rate_df = (
    heirarchy_mapping_df
    .withColumn("rnk", F.dense_rank().over(Window.partitionBy(su.normalize_text(F.col("LOWEST_LEVEL_PORTFOLIO_NAME"))).orderBy(F.col("REPORTING_DATE_KEY").desc())))
    .filter( F.col("rnk") == 1 )
    .select(
        su.normalize_text(F.col("LOWEST_LEVEL_PORTFOLIO_NAME")).alias("LLP_KEY"),
        F.col("TAX_RATE").cast("double").alias("TAX_RATE")
    )
    .dropDuplicates(["LLP_KEY"])
)

dv01_with_llp_df = (
    total_dv01_df.alias("d")
    .join(
        llp_mapping_df.alias("m"),
        on = ["PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY"],
        how = "left"
    )
    .select("d.*", "m.LOWEST_LEVEL_PORTFOLIO_NAME")
)
validate_lookup(dv01_with_llp_df, "LOWEST_LEVEL_PORTFOLIO_NAME", ["PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY"], "LLP mapping")

dv01_with_tax_df = (
    dv01_with_llp_df.alias("d")
    .join(
        tax_rate_df.alias("t"),
        on = su.normalize_text(F.col("d.LOWEST_LEVEL_PORTFOLIO_NAME")) == F.col("t.LLP_KEY"),
        how = "left"
    )
    .select("d.*", "t.TAX_RATE")
)
validate_lookup(dv01_with_tax_df, "TAX_RATE", ["LOWEST_LEVEL_PORTFOLIO_NAME"], "Tax rate")

display(dv01_with_tax_df)

# COMMAND ----------

# ============================================================
# FX RATE LOOKUP BY DV01 REPORTING MONTH
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ------------------------------------------------------------
# 1. Configured FX column names
# ------------------------------------------------------------

fx_currency_col = getattr(
    dataset_cfg,
    "fx_currency_col",
    "CODE"
)

fx_rate_col = getattr(
    dataset_cfg,
    "fx_rate_col",
    "BS_RATE"
)

fx_date_col = getattr(
    dataset_cfg,
    "fx_date_col",
    "DATE"
)


# ------------------------------------------------------------
# 2. Confirm DV01 REPORT_DATE is based on INCEPTION_DATE month
# ------------------------------------------------------------
#
# Expected:
# INCEPTION_DATE = 2026-05-01
# REPORT_DATE    = 2026-05-31
# ------------------------------------------------------------

dv01_required_months_df = (
    dv01_with_tax_df
    .select(
        F.col("REPORT_DATE").cast("date").alias("REPORT_DATE")
    )
    .filter(
        F.col("REPORT_DATE").isNotNull()
    )
    .distinct()
)


print("DV01 reporting months required for the FX lookup:")

display(
    dv01_required_months_df
    .orderBy("REPORT_DATE")
)


# ------------------------------------------------------------
# 3. Parse and normalize the FX source
# ------------------------------------------------------------
#
# Source format:
# 31-May-26
# 31-Jul-26
#
# Spark pattern:
# dd-MMM-yy
# ------------------------------------------------------------

fx_rates_norm = (
    fx_rates_df

    .withColumn(
        "__FX_SOURCE_DATE",
        F.to_date(
            F.trim(
                F.col(fx_date_col).cast("string")
            ),
            "dd-MMM-yy"
        )
    )

    .withColumn(
        "REPORT_DATE",
        F.last_day(
            F.col("__FX_SOURCE_DATE")
        )
    )

    .withColumn(
        "__FX_CURRENCY",
        F.upper(
            F.trim(
                F.col(fx_currency_col).cast("string")
            )
        )
    )

    .withColumn(
        "__FX_RATE",
        F.col(fx_rate_col).cast("double")
    )
)


# ------------------------------------------------------------
# 4. Validate FX date conversion
# ------------------------------------------------------------

invalid_fx_dates_df = (
    fx_rates_norm

    .filter(
        F.col(fx_date_col).isNotNull()
        & F.col("__FX_SOURCE_DATE").isNull()
    )

    .select(
        F.col(fx_date_col).alias("ORIGINAL_FX_DATE")
    )

    .distinct()
)


invalid_fx_date_count = invalid_fx_dates_df.count()


if invalid_fx_date_count > 0:
    invalid_fx_dates = [
        row.asDict()
        for row in invalid_fx_dates_df.collect()
    ]

    raise ValueError(
        f"Unable to parse {invalid_fx_date_count} distinct FX date value(s). "
        f"Expected the format dd-MMM-yy, such as 31-May-26. "
        f"Invalid values: {invalid_fx_dates}"
    )


# ------------------------------------------------------------
# 5. Show all available FX months
# ------------------------------------------------------------

print("FX months available in fx_rates_df:")

display(
    fx_rates_norm

    .select(
        "REPORT_DATE"
    )

    .filter(
        F.col("REPORT_DATE").isNotNull()
    )

    .distinct()

    .orderBy(
        "REPORT_DATE"
    )
)


# ------------------------------------------------------------
# 6. Show USD records available in the source
# ------------------------------------------------------------
#
# This check must show a 2026-05-31 USD record if May USD FX is
# available in the DataFrame loaded by this notebook.
# ------------------------------------------------------------

print("USD FX records available in fx_rates_df:")

display(
    fx_rates_norm

    .filter(
        F.col("__FX_CURRENCY") == "USD"
    )

    .select(
        F.col(fx_date_col).alias("ORIGINAL_DATE"),
        "__FX_SOURCE_DATE",
        "REPORT_DATE",
        "__FX_CURRENCY",
        "__FX_RATE"
    )

    .orderBy(
        "REPORT_DATE"
    )
)


# ------------------------------------------------------------
# 7. Restrict FX records to months required by DV01
# ------------------------------------------------------------
#
# This removes July FX records when the DV01 data requires May.
#
# It is not the final join. It is a safe pre-filter that ensures
# only relevant reporting months are retained.
# ------------------------------------------------------------

fx_rates_required_months_df = (
    fx_rates_norm.alias("f")

    .join(
        F.broadcast(
            dv01_required_months_df.alias("m")
        ),

        on=(
            F.col("f.REPORT_DATE")
            == F.col("m.REPORT_DATE")
        ),

        how="inner"
    )

    .select(
        "f.*"
    )
)


print("FX records retained for the required DV01 months:")

display(
    fx_rates_required_months_df

    .select(
        F.col(fx_date_col).alias("ORIGINAL_DATE"),
        "__FX_SOURCE_DATE",
        "REPORT_DATE",
        "__FX_CURRENCY",
        "__FX_RATE"
    )

    .orderBy(
        "REPORT_DATE",
        "__FX_CURRENCY"
    )
)


# ------------------------------------------------------------
# 8. Keep one FX rate per currency and reporting month
# ------------------------------------------------------------
#
# We do not select the latest FX rate across all months.
#
# The ranking is within:
#   currency + reporting month
#
# Therefore, May and July remain completely separate.
# ------------------------------------------------------------

fx_month_window = (
    Window

    .partitionBy(
        F.col("__FX_CURRENCY"),
        F.col("REPORT_DATE")
    )

    .orderBy(
        F.col("__FX_SOURCE_DATE").desc()
    )
)


fx_rate_df = (
    fx_rates_required_months_df

    .withColumn(
        "__FX_MONTH_RANK",
        F.row_number().over(
            fx_month_window
        )
    )

    .filter(
        F.col("__FX_MONTH_RANK") == 1
    )

    .select(
        F.col("__FX_CURRENCY").alias("FX_CURRENCY"),
        F.col("REPORT_DATE").alias("FX_REPORT_DATE"),
        F.col("__FX_RATE").alias("FX_RATE")
    )
)


print("Final FX lookup table by currency and reporting month:")

display(
    fx_rate_df

    .orderBy(
        "FX_REPORT_DATE",
        "FX_CURRENCY"
    )
)


# ------------------------------------------------------------
# 9. Join using currency AND reporting month
# ------------------------------------------------------------
#
# A May DV01 record can only match a May FX record.
# A July FX record cannot match a May DV01 record.
# ------------------------------------------------------------

dv01_with_fx_df = (
    dv01_with_tax_df.alias("d")

    .join(
        fx_rate_df.alias("f"),

        on=(
            (
                F.upper(
                    F.trim(
                        F.col("d.SECURITY_CURRENCY")
                    )
                )
                ==
                F.col("f.FX_CURRENCY")
            )
            &
            (
                F.col("d.REPORT_DATE").cast("date")
                ==
                F.col("f.FX_REPORT_DATE").cast("date")
            )
        ),

        how="left"
    )

    .select(
        "d.*",
        F.col("f.FX_REPORT_DATE"),
        F.col("f.FX_RATE")
    )
)


# ------------------------------------------------------------
# 10. Prove that DV01 month and FX month are identical
# ------------------------------------------------------------

date_mismatch_count = (
    dv01_with_fx_df

    .filter(
        F.col("FX_RATE").isNotNull()
        &
        (
            F.col("REPORT_DATE")
            != F.col("FX_REPORT_DATE")
        )
    )

    .count()
)


if date_mismatch_count > 0:
    raise ValueError(
        f"Found {date_mismatch_count} record(s) where the DV01 REPORT_DATE "
        f"does not match the FX_REPORT_DATE."
    )


# ------------------------------------------------------------
# 11. Display the result with both dates temporarily
# ------------------------------------------------------------
#
# For a May inception month, this should show:
#
# REPORT_DATE    = 2026-05-31
# FX_REPORT_DATE = 2026-05-31
# ------------------------------------------------------------

print("DV01 to FX matching result:")

display(
    dv01_with_fx_df

    .select(
        "REPORT_DATE",
        "FX_REPORT_DATE",
        "SECURITY_CURRENCY",
        "FX_RATE"
    )

    .distinct()

    .orderBy(
        "REPORT_DATE",
        "SECURITY_CURRENCY"
    )
)


# ------------------------------------------------------------
# 12. Identify missing FX records
# ------------------------------------------------------------

missing_fx_df = (
    dv01_with_fx_df

    .filter(
        F.col("FX_RATE").isNull()
    )

    .select(
        "SECURITY_CURRENCY",
        "REPORT_DATE"
    )

    .distinct()

    .orderBy(
        "REPORT_DATE",
        "SECURITY_CURRENCY"
    )
)


missing_fx_count = missing_fx_df.count()


if missing_fx_count > 0:
    print(
        f"FX rate lookup failed for {missing_fx_count} distinct "
        f"currency and reporting-month combination(s)."
    )

    display(
        missing_fx_df
    )


# ------------------------------------------------------------
# 13. Final validation
# ------------------------------------------------------------
#
# Do not fill missing FX rates with 1.0 before this validation.
# A default of 1.0 would hide the fact that the required May
# record is unavailable.
# ------------------------------------------------------------

validate_lookup(
    dv01_with_fx_df,
    "FX_RATE",
    [
        "SECURITY_CURRENCY",
        "REPORT_DATE"
    ],
    "FX rate"
)


# ------------------------------------------------------------
# 14. Remove temporary FX date after successful validation
# ------------------------------------------------------------

dv01_with_fx_df = (
    dv01_with_fx_df
    .drop(
        "FX_REPORT_DATE"
    )
)


display(
    dv01_with_fx_df
)

# COMMAND ----------

partial_dv01_df = (
    dv01_with_fx_df
    .withColumn(
        "ASSET_HOLDING",
        F.when( F.col("PROGRAM_CODE").isin("GH_14", "GH_09", "GH_02"), "Bond" )
         .when( F.col("PROGRAM_CODE") == "GH_01", "Liability" )
         .when( F.col("PROGRAM_CODE") == "MB_01", "Manubank" )
    )
    .withColumn("SUM_PARTIAL_DV01", F.col("TOTAL_DV01") * F.col("FX_RATE") * (1 - F.col("TAX_RATE")))
    .select(
        "REPORT_DATE", "SECURITY_CURRENCY", "ASSET_HOLDING", "TENOR_CD", "TERM_YEARS",
        "PROGRAM_CODE", "LOWEST_LEVEL_PORTFOLIO_NAME", "LEGAL_ID",
        "TOTAL_DV01", "FX_RATE", "TAX_RATE", "SUM_PARTIAL_DV01"
    )
)
validate_lookup(partial_dv01_df, "ASSET_HOLDING", ["PROGRAM_CODE"], "Asset holding")

if partial_dv01_df.count() != total_dv01_df.count():
    raise ValueError("Row count changed across the lookups - a mapping has duplicate keys")



# COMMAND ----------

dv01_unpivot_df = dv01_unpivot_df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
#uc.write(dv01_unpivot_df, dataset_cfg.unpivot_table, dataset_cfg.unpivot_ext_table_loc)
display(dv01_unpivot_df)
partial_dv01_df = partial_dv01_df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
#uc.write(partial_dv01_df, dataset_cfg.curation_table, dataset_cfg.curation_ext_table_loc)
display(partial_dv01_df)

# COMMAND ----------

logger.info("HA Ineffectiveness - Partial DV01 curation loading completed")