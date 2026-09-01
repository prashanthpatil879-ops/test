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
heirarchy_mapping_df = uc.read_latest(dataset_cfg.inputs.heirarchy_mapping_latest)

# read_table, NOT read_latest. Each monthly IFS fetch appends exactly one month of
# rates, so the newest INGESTION_TS carries only that month (the 2026-09-01 fetch
# holds nothing but 31-Aug-26). read_latest would therefore discard every prior
# month and leave nothing to match an April or June REPORT_DATE against.
fx_rates_df = uc.read_table(dataset_cfg.inputs.fx_rates)

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
    """Tenor bucket columns to rows. TENOR_CD is emitted in YEARS (003M -> 0.25)."""
    tenors = [c for c in df.columns if re.fullmatch(r"\d+M", c)]

    if not tenors:
        raise ValueError(f"No tenor columns found in: {df.columns}")

    logger.info(f"Unpivoting {len(tenors)} tenor columns to years: { {c: int(c[:-1]) / 12 for c in tenors} }")
    expr = "stack({0}, {1}) as (TENOR_CD, DV01_VALUE)".format(
        len(tenors), ", ".join([f"CAST({int(c[:-1]) / 12} AS DOUBLE), CAST(`{c}` AS DOUBLE)" for c in tenors])
    )
    return df.select("ASSET_TYPE", "INCEPTION_DATE", "PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY", "SOURCE", F.expr(expr))


def validate_lookup(df, lookup_col, key_cols, source):
    """Warn on unmatched keys and carry on. Matched rows are unaffected."""
    unmatched = df.filter(F.col(lookup_col).isNull()).select(*key_cols).distinct().collect()

    if unmatched:
        logger.warning(
            f"{source} lookup: no match for {len(unmatched)} key(s) {[r.asDict() for r in unmatched]}. "
            f"These rows keep a null {lookup_col} and a null SUM_PARTIAL_DV01; all other rows are unaffected."
        )
    else:
        logger.info(f"{source} lookup matched every row")

    return unmatched

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

# REPORT_DATE: ALM stamps a month's data on the 1st of the FOLLOWING month, so the
# reporting date is the last day of the PREVIOUS month.
#   INCEPTION_DATE 2026-05-01 -> REPORT_DATE 2026-04-30
#   INCEPTION_DATE 2026-07-01 -> REPORT_DATE 2026-06-30  (Q2 close)
#   INCEPTION_DATE 2026-08-01 -> REPORT_DATE 2026-07-31
# add_months(-1) is required: F.last_day(INCEPTION_DATE) on its own returns 2026-05-31.
#
# TOTAL_DV01 = sum of DV01 across the three ALM datasets, per tenor, grouped on
# PROGRAM_CODE + LEGAL_ID + SECURITY_CURRENCY. The datasets are disjoint on
# PROGRAM_CODE (GH_01 only in liab_post_hp + derivatives, the rest only in
# fi + derivatives), so this reproduces the per-combination source rules in the
# "Calculation sum_partial_dv01" column of HA DV01 mapping_NT_LLP.xlsx.
total_dv01_df = (
    dv01_unpivot_df
    .withColumn("REPORT_DATE", F.last_day(F.add_months(F.col("INCEPTION_DATE"), -1)))
    .groupBy("REPORT_DATE", "PROGRAM_CODE", "LEGAL_ID", "SECURITY_CURRENCY", "TENOR_CD")
    .agg( F.sum("DV01_VALUE").alias("TOTAL_DV01") )
)

logger.info(f"Report date(s) derived from INCEPTION_DATE: {[r.REPORT_DATE for r in total_dv01_df.select('REPORT_DATE').distinct().collect()]}")
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

# FX rate is matched on BOTH currency and date. The IFS file carries month-end
# dates as '30-Jun-26' while the ALM side derives 2026-06-30, so the string is
# parsed to a real date and joined on REPORT_DATE. This tracks forward on its own:
# INCEPTION_DATE 2026-08-01 -> REPORT_DATE 2026-07-31 -> FX row '31-Jul-26'.
fx_currency_col = dataset_cfg.fx_currency_col
fx_rate_col = dataset_cfg.fx_rate_col
fx_date_col = dataset_cfg.fx_date_col
fx_date_fmt = dataset_cfg.fx_date_fmt

fx_date_dtype = dict(fx_rates_df.dtypes)[fx_date_col]
fx_date_expr = (
    F.col(fx_date_col).cast("date") if fx_date_dtype in ("date", "timestamp")
    else F.try_to_date(F.col(fx_date_col), fx_date_fmt)
)

fx_rate_df = (
    fx_rates_df
    .withColumn("FX_CURRENCY", F.upper(F.trim(F.col(fx_currency_col))))
    .withColumn("FX_DATE", fx_date_expr)
    .filter( F.col("FX_DATE").isNotNull() )
    # The same CODE + DATE is re-published by several monthly fetches, so keep the
    # most recently ingested copy of each.
    .withColumn("rnk", F.dense_rank().over(Window.partitionBy("FX_CURRENCY", "FX_DATE").orderBy(F.col("INGESTION_TS").desc())))
    .filter( F.col("rnk") == 1 )
    .select("FX_CURRENCY", "FX_DATE", F.col(fx_rate_col).cast("double").alias("FX_RATE"))
    .dropDuplicates(["FX_CURRENCY", "FX_DATE"])
)

unparsed_fx_dates = fx_rates_df.filter(fx_date_expr.isNull()).count()
if unparsed_fx_dates:
    logger.warning(f"{unparsed_fx_dates} FX row(s) have a {fx_date_col} that does not parse as '{fx_date_fmt}' and were dropped.")

display(fx_rate_df)

dv01_with_fx_df = (
    dv01_with_tax_df.alias("d")
    .join(
        fx_rate_df.alias("f"),
        on = [
            F.col("d.SECURITY_CURRENCY") == F.col("f.FX_CURRENCY"),
            F.col("d.REPORT_DATE") == F.col("f.FX_DATE")
        ],
        how = "left"
    )
    .select("d.*", "f.FX_RATE")
)

# A CAD-based rates file has no row for CAD itself, so the reporting currency is
# given a rate of 1 explicitly. This is deliberately scoped to base_currency only:
# every OTHER unmatched currency stays null and is reported by validate_lookup,
# because defaulting e.g. JPY to 1.0 would silently misstate it by ~100x.
base_currency = dataset_cfg.get("base_currency", None)
if base_currency:
    base_filled = dv01_with_fx_df.filter( F.col("FX_RATE").isNull() & (F.col("SECURITY_CURRENCY") == F.lit(base_currency)) ).count()
    if base_filled:
        logger.warning(f"{base_filled} row(s) in the reporting currency {base_currency} had no FX row; applying FX_RATE = 1.0.")
        dv01_with_fx_df = dv01_with_fx_df.withColumn(
            "FX_RATE",
            F.when( F.col("FX_RATE").isNull() & (F.col("SECURITY_CURRENCY") == F.lit(base_currency)), F.lit(1.0) )
             .otherwise( F.col("FX_RATE") )
        )

validate_lookup(dv01_with_fx_df, "FX_RATE", ["REPORT_DATE", "SECURITY_CURRENCY"], "FX rate")

display(dv01_with_fx_df)

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
        "REPORT_DATE", "SECURITY_CURRENCY", "ASSET_HOLDING", "TENOR_CD",
        "PROGRAM_CODE", "LOWEST_LEVEL_PORTFOLIO_NAME", "LEGAL_ID",
        "TOTAL_DV01", "FX_RATE", "TAX_RATE", "SUM_PARTIAL_DV01"
    )
)
validate_lookup(partial_dv01_df, "ASSET_HOLDING", ["PROGRAM_CODE"], "Asset holding")

# Left joins never drop rows, so the grain must still equal TOTAL_DV01.
# A mismatch means a mapping has duplicate keys and DV01 has been double counted.
if partial_dv01_df.count() != total_dv01_df.count():
    raise ValueError("Row count changed across the lookups - a mapping has duplicate keys")

incomplete = partial_dv01_df.filter(F.col("SUM_PARTIAL_DV01").isNull()).count()
if incomplete:
    logger.warning(
        f"{incomplete} of {partial_dv01_df.count()} rows have a null SUM_PARTIAL_DV01 because a lookup "
        f"did not match. Review the warnings above before using this output downstream."
    )

display(partial_dv01_df)

# COMMAND ----------

dv01_unpivot_df = dv01_unpivot_df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
uc.write(dv01_unpivot_df, dataset_cfg.unpivot_table, dataset_cfg.unpivot_ext_table_loc)

partial_dv01_df = partial_dv01_df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
uc.write(partial_dv01_df, dataset_cfg.curation_table, dataset_cfg.curation_ext_table_loc)

# COMMAND ----------

logger.info("HA Ineffectiveness - Partial DV01 curation loading completed")
