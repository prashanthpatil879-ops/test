from pyspark.sql import DataFrame
from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType, DecimalType
import pandas as pd
from typing import NamedTuple
from rda.engine.base_step import RDABaseStep
import rda.utils.spark_utils as su
from datetime import datetime


class SCAInputs(NamedTuple):
    st_propogation_df: DataFrame
    country_currency_mapping_df: DataFrame
    market_scenario_df: DataFrame
    factor_scenario_df: DataFrame


class SCA(RDABaseStep):

    def __init__(self, env_config, spark, step_config, analysis_date):
        super().__init__(env_config, spark, step_config, analysis_date)

    def read(self):
        inputs = self._step_config.inputs

        if not self._valuation_date or str(self._valuation_date).lower() == 'yyyy-mm-dd':
            msg = f"Analysis date required in the format YYYY-MM-DD"
            self._logger.error(msg)
            raise ValueError(msg)
        else:
            try:
                validated_date = datetime.strptime(self._valuation_date, "%Y-%m-%d").date()
                self._logger.info(f"Detected supplied analysis date: {validated_date}")
            except ValueError:
                msg = f"Invalid analysis date '{self._valuation_date}'. Please enter a date in YYYY-MM-DD format."
                self._logger.error(msg)
                raise ValueError(msg)

        self._logger.info(f"Reading inputs for SCA for analysis date: {self._valuation_date}")

        input_dfs = SCAInputs(
            st_propogation_df = self._uc.read_latest(inputs.st_propogation.table_name, filter=f"to_date(ANALYSIS_DT)='{self._valuation_date}' and CURRENT_VERSION='Y'"),
            country_currency_mapping_df = self._uc.read_latest(inputs.country_currency_mapping.table_name),
            market_scenario_df = self._uc.read_latest(inputs.market_scenario.table_name),
            factor_scenario_df = self._uc.read_latest(inputs.factor_scenario.table_name)
        )        
        self.validate_inputs(input_dfs)
        return input_dfs    
    

    def curate_st_propogation(self, inputs):
        self._logger.info("Deriving additional columns for ST Propogation...")

        window = Window.partitionBy("COUNTRY").orderBy( F.when(F.col("CURRENCY_CODE").isin(["EUR", "NZD", "GBP", "DKK", "AUD"]), 0).otherwise(1) )
        country_currency_dedup_df = inputs.country_currency_mapping_df.withColumn("RANK", F.dense_rank().over(window)).filter(F.col("RANK") == 1).drop("RANK")

        st_propogation_base_df = (
            inputs.st_propogation_df
            .withColumn("SCENARIO_TYPE", F.lit("BarraOne St Propogation"))
            .withColumn(
                "RISK_FACTOR_SUBTYPE",
                F.when( (F.col("ASSET_CLASS_NM") == "Common Factor") & (F.col("FACTOR_GROUP_NM") == "Term Structure" ), "RF" )
                 .when( (F.col("ASSET_CLASS_NM") == "Common Factor") & (F.col("FACTOR_GROUP_NM") == "Spread") & (F.col("FACTOR_SUBGROUP_NM").contains("Corporate") ), "CS" )
                 .when( (F.col("ASSET_CLASS_NM") == "Common Factor") & (F.col("FACTOR_GROUP_NM") == "Spread") & (F.col("FACTOR_SUBGROUP_NM").contains("Swap") ), "SS" )
                 .when( (F.col("ASSET_CLASS_NM") == "Currency"), "Curr" ) 
                 .when( (F.col("ASSET_CLASS_NM") == "Common Factor") & (F.col("FACTOR_GROUP_NM").isin(["Country", "Market"])), "PE" )
                .otherwise("Unclassified")
            )
            .withColumn(
                "SCALED_SHOCK",
                F.when(
                    F.col("RISK_FACTOR_SUBTYPE").isin(["RF", "SS", "CS"]),
                    F.col("FACTOR_SHOCK_AMT") * -1
                ).otherwise(F.lit(None))
            )
            .withColumn("NORMALIZED_BPS_SHOCK", F.col("SCALED_SHOCK") * 100)
            .withColumn(
                "TENOR",
                F.when(
                    F.col("RISK_FACTOR_SUBTYPE") == "RF",
                    F.regexp_extract(F.col("FACTOR_SUBGROUP_NM"), r"^[A-Z]{3} Rate ([A-Z0-9]+)$", 1)
                ).otherwise(F.lit(None))
            )
        )

        pe_records_df = st_propogation_base_df.filter(F.col("RISK_FACTOR_SUBTYPE") == "PE")
        pe_records_df = pe_records_df.withColumn("COUNTRY_TEMP", F.regexp_extract(F.col("FACTOR_SUBGROUP_NM"), r"^[A-Z]+\s(.*)\sMkt$", 1))
        pe_records_df = pe_records_df.withColumn("COUNTRY_SHORT_TEMP", F.regexp_extract(F.col("FACTOR_SUBGROUP_NM"), r"^([A-Z]+)\sCountry$", 1))

        pe_records_df = (
            pe_records_df.alias("pe")
            .join(
                country_currency_dedup_df.alias("c"),
                on = su.normalize_text_without_space(F.col("pe.COUNTRY_TEMP")) == su.normalize_text_without_space(F.col("c.COUNTRY")),
                how = "left"
            )
            .join(
                country_currency_dedup_df.alias("c2"),
                on = su.normalize_text_without_space(F.col("pe.COUNTRY_SHORT_TEMP")) == su.normalize_text_without_space(F.col("c2.TWO_LETTER_COUNTRY_CODE")),
                how = "left"
            )
            .select( "pe.*", F.coalesce(F.col("c.CURRENCY_CODE"), F.col("c2.CURRENCY_CODE")).alias("CURRENCY_CODE") )
        )
        pe_records_df = (
            pe_records_df
            .withColumn(
                "CURRENCY_CODE",
                F.when( F.col("FACTOR_SUBGROUP_NM").contains("Emerging Markets"), F.lit("EMM") )
                .otherwise(F.col("CURRENCY_CODE"))
            ) 
        )

        cs_records_df = st_propogation_base_df.filter(F.col("RISK_FACTOR_SUBTYPE") == "CS")
        cs_records_df = cs_records_df.withColumn("COUNTRY_TEMP", F.regexp_extract(F.col("FACTOR_SUBGROUP_NM"), r"^([A-Z]+)\s", 1))

        cs_records_df = (
            cs_records_df.alias("cs")
            .join(
                country_currency_dedup_df.alias("c"),
                on = su.normalize_text_without_space(F.col("cs.COUNTRY_TEMP")) == su.normalize_text_without_space(F.col("c.TWO_LETTER_COUNTRY_CODE")),
                how = "left"
            )
            .select( "cs.*", F.coalesce(F.col("c.CURRENCY_CODE"), F.lit(None)).alias("CURRENCY_CODE") )
        )

        ss_records_df = st_propogation_base_df.filter(F.col("RISK_FACTOR_SUBTYPE") == "SS")
        ss_records_df = ss_records_df.withColumn("CURR_TEMP", F.regexp_extract(F.col("FACTOR_SUBGROUP_NM"), r"^([A-Z]+)\s", 1))

        ss_records_df = (
            ss_records_df.alias("ss")
            .join(
                country_currency_dedup_df.alias("c"),
                on = su.normalize_text_without_space(F.col("ss.CURR_TEMP")) == su.normalize_text_without_space(F.col("c.CURRENCY_CODE")),
                how = "left"
            )
            .select( "ss.*", F.coalesce(F.col("c.CURRENCY_CODE"), F.lit(None)).alias("CURRENCY_CODE") )
        )

        curr_records_df = st_propogation_base_df.filter(F.col("RISK_FACTOR_SUBTYPE") == "Curr")

        curr_records_df = (
            curr_records_df.alias("curr")
            .join(
                country_currency_dedup_df.alias("c"),
                on = su.normalize_text_without_space(F.col("curr.FACTOR_GROUP_NM")) == su.normalize_text_without_space(F.col("c.CURRENCY")),
                how = "left"
            )
            .select( "curr.*", F.coalesce(F.col("c.CURRENCY_CODE"), F.lit(None)).alias("CURRENCY_CODE") )
        )

        rf_records_df = st_propogation_base_df.filter(F.col("RISK_FACTOR_SUBTYPE") == "RF")
        rf_records_df = rf_records_df.withColumn("CURR_TEMP", F.regexp_extract(F.col("FACTOR_SUBGROUP_NM"), r"^([A-Z]+)\sRate", 1))

        rf_records_df = (
            rf_records_df.alias("rf")
            .join(
                country_currency_dedup_df.alias("c"),
                on = su.normalize_text_without_space(F.col("rf.CURR_TEMP")) == su.normalize_text_without_space(F.col("c.CURRENCY_CODE")),
                how = "left"
            )
            .select( "rf.*", F.coalesce(F.col("c.CURRENCY_CODE"), F.lit(None)).alias("CURRENCY_CODE") )
        )

        st_propogation_derived_df = (
            pe_records_df.unionByName(cs_records_df, allowMissingColumns=True)
            .unionByName(ss_records_df, allowMissingColumns=True)
            .unionByName(curr_records_df, allowMissingColumns=True)
            .unionByName(rf_records_df, allowMissingColumns=True)
            .unionByName(st_propogation_base_df.filter(F.col("RISK_FACTOR_SUBTYPE") == "Unclassified"), allowMissingColumns=True )
            .select("ST_PROPAGATION_KEY", "ASSET_CLASS_NM", F.col("ANALYSIS_DT").cast("date"), "FACTOR_GROUP_NM", "FACTOR_SUBGROUP_NM", "STRESS_SCENARIO_NM", "ACTIVE_FACTOR_EXPOSURE_AMT", "BENCHMARK_FACTOR_EXPOSURE_AMT","PORTFOLIO_FACTOR_EXPOSURE_AMT", "FACTOR_PL_CONTRIBUTION_PCT", "ACTIVE_FACTOR_PL_CONTRIBUTION_PCT", "BENCHMARK_FACTOR_PL_CONTRIBUTION_PCT", "FACTOR_SHOCK_AMT", "SCENARIO_TYPE", "TENOR", "RISK_FACTOR_SUBTYPE", "SCALED_SHOCK", "NORMALIZED_BPS_SHOCK", "CURRENCY_CODE")
        )               

        st_propogation_derived_df = self.cache_df(st_propogation_derived_df, "st_propogation_derived_df")

        self.write(st_propogation_derived_df, self._step_config.step_st_propogation.table_name, self._step_config.step_st_propogation.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info(f"ST Propogation curation completed")

        return st_propogation_derived_df
    

    def calculate_unified_shocks(self, inputs, st_propogation_derived_df):

        self._logger.info(f"Calculating unified shocks...")
        
        market_scenario_df = (
            inputs.market_scenario_df
            .select("SCENARIO_NAME", "SCENARIO_TYPE", "SCENARIO_SUBTYPE", "NORMALIZED_BPS_SHOCK", "RISK_FACTOR_SUBTYPE", "CURRENCY_CODE", F.upper(F.col("SHOCK_VARIABLE")).alias("TENOR"))
        )

        factor_scenario_df = (
            inputs.factor_scenario_df
            .select("SCENARIO_NAME", "SCENARIO_TYPE", F.lit(None).alias("SCENARIO_SUBTYPE"), "NORMALIZED_BPS_SHOCK", "RISK_FACTOR_SUBTYPE", "CURRENCY_CODE", F.upper(F.col("TENOR")) )
        )

        st_propogation_df = (
            st_propogation_derived_df
            .select(F.col("STRESS_SCENARIO_NM").alias("SCENARIO_NAME"), "SCENARIO_TYPE", F.lit(None).alias("SCENARIO_SUBTYPE"), "NORMALIZED_BPS_SHOCK", "RISK_FACTOR_SUBTYPE", "CURRENCY_CODE", F.upper(F.col("TENOR")) )
        )

        base_df = market_scenario_df \
        .union(factor_scenario_df) \
        .union(st_propogation_df) \
        .filter(
            (
                (F.col("RISK_FACTOR_SUBTYPE").isin(["RF", "SS"]))
                &
                (F.col("TENOR").isin(["2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]))
            )
            |
            F.col("RISK_FACTOR_SUBTYPE").isin(["PE", "Curr", "CS"])
        )

        es_sensitivity_base_df = (
            base_df
            .withColumn(
                "ES_SENSITIVITY_NAME",
                F.when(
                    F.col("RISK_FACTOR_SUBTYPE") == "RF",
                    F.when( F.col("NORMALIZED_BPS_SHOCK") <= -100, F.concat(F.lit("RF -100bps "), F.col("TENOR"), F.lit(" Only")) ) 
                    .when( (F.col("NORMALIZED_BPS_SHOCK") > -100) & (F.col("NORMALIZED_BPS_SHOCK") < -50), F.concat(F.lit("RF -100bps-50bps "), F.col("TENOR"), F.lit(" Only")) )
                    .when( (F.col("NORMALIZED_BPS_SHOCK") >= -50) & (F.col("NORMALIZED_BPS_SHOCK") < 0), F.concat(F.lit("RF -50bps "), F.col("TENOR"), F.lit(" Only")) )
                    .when( (F.col("NORMALIZED_BPS_SHOCK") >= 0) & (F.col("NORMALIZED_BPS_SHOCK") <= 50), F.concat(F.lit("RF +50bps "), F.col("TENOR"), F.lit(" Only")) )
                    .when( (F.col("NORMALIZED_BPS_SHOCK") > 50) & (F.col("NORMALIZED_BPS_SHOCK") < 100), F.concat(F.lit("RF +100bps+50bps "), F.col("TENOR"), F.lit(" Only")) )
                    .when( F.col("NORMALIZED_BPS_SHOCK") >= 100, F.concat(F.lit("RF +100bps "), F.col("TENOR"), F.lit(" Only")) )
                )
                .when(
                    F.col("RISK_FACTOR_SUBTYPE") == "SS",
                    F.when( F.col("NORMALIZED_BPS_SHOCK") < 0, F.lit("SS -20bps") )
                    .otherwise(F.lit("SS +20bps"))
                )
                .when(
                    F.col("RISK_FACTOR_SUBTYPE") == "CS",
                    F.when( F.col("NORMALIZED_BPS_SHOCK") < 0, F.lit("CS -50bps") )
                    .otherwise(F.lit("CS +50bps"))
                )
                .otherwise(F.lit(None))
            )
        )

        rf_neg100_df = (
            es_sensitivity_base_df
            .filter(F.col("ES_SENSITIVITY_NAME").contains("-100bps-50bps"))
            .withColumn( "ES_SENSITIVITY_NAME", F.regexp_replace( F.col("ES_SENSITIVITY_NAME"), r"\-50bps", "") )
        )

        rf_pos100_df = (
            es_sensitivity_base_df
            .filter( F.col("ES_SENSITIVITY_NAME").contains("+100bps+50bps") )
            .withColumn( "ES_SENSITIVITY_NAME", F.regexp_replace( F.col("ES_SENSITIVITY_NAME"), r"\+50bps", "") )
        )

        es_sensitivity_df = (
            es_sensitivity_base_df
            .withColumn(
                "ES_SENSITIVITY_NAME",
                F.regexp_replace( F.regexp_replace( F.col("ES_SENSITIVITY_NAME"), r"RF \-100bps\-50bps", "RF -50bps"), r"RF \+100bps\+50bps", "RF +50bps")
            )
            .union(rf_neg100_df)
            .union(rf_pos100_df)
            .withColumn("ABS_SHOCK_BPS", F.abs(F.col("NORMALIZED_BPS_SHOCK")))
        )

        es_sensitivity_sf_df = (
            es_sensitivity_df
            .withColumn(
                "SHOCK_FACTOR",
                F.when(
                    F.col("RISK_FACTOR_SUBTYPE") == "RF",
                    F.when( F.col("ABS_SHOCK_BPS") <= 50, F.col("ABS_SHOCK_BPS") / 50 )
                    .when( F.col("ABS_SHOCK_BPS") >= 100, F.col("ABS_SHOCK_BPS") / 100 )
                    .when( F.col("ES_SENSITIVITY_NAME").contains("50bps"), F.abs((100 - F.col("ABS_SHOCK_BPS")) / 50) )
                    .when( F.col("ES_SENSITIVITY_NAME").contains("100bps"), F.abs((F.col("ABS_SHOCK_BPS") - 50) / 50) )
                )
                .when(
                    F.col("RISK_FACTOR_SUBTYPE") == "SS",
                    F.col("ABS_SHOCK_BPS") / 20
                )
                .when(
                    F.col("RISK_FACTOR_SUBTYPE") == "CS",
                    F.col("ABS_SHOCK_BPS") / 50
                )
            )
            .drop("ABS_SHOCK_BPS")
        )

        curr_main_records = (
            es_sensitivity_sf_df
            .filter( (F.col("RISK_FACTOR_SUBTYPE") == "Curr") )
            .distinct()
        )

        curr_cad_records = (
            curr_main_records
            .filter( F.col("CURRENCY_CODE") == "CAD" )
            .select("SCENARIO_NAME", "SCENARIO_TYPE", "NORMALIZED_BPS_SHOCK")
            .distinct()
        )

        curr_sf_df = (
            curr_main_records.alias("curr")
            .join(
                curr_cad_records.alias("cad"),
                on = [
                    F.col("curr.SCENARIO_NAME") == F.col("cad.SCENARIO_NAME"),
                    F.col("curr.SCENARIO_TYPE") == F.col("cad.SCENARIO_TYPE")
                ],
                how = "left"
            )
            .select("curr.*", F.col("cad.NORMALIZED_BPS_SHOCK").alias("CAD_NORMALIZED_BPS_SHOCK"))
            .withColumn(
                "SHOCK_FACTOR",
                F.when(
                    F.col("CURRENCY_CODE") == "CAD", F.lit(0)
                )
                .otherwise(
                    (
                        1 + F.col("NORMALIZED_BPS_SHOCK") / 100
                    )
                    /
                    (
                        1 + F.col("CAD_NORMALIZED_BPS_SHOCK") / 100
                    )
                    - 1
                )
            )
            .drop("CAD_NORMALIZED_BPS_SHOCK")
        )

        unified_shocks_base_df = (
            es_sensitivity_sf_df
            .filter( F.col("RISK_FACTOR_SUBTYPE") != "Curr" )
            .union(curr_sf_df)
        )

        pe_main_df = unified_shocks_base_df.filter( F.col("RISK_FACTOR_SUBTYPE") == "PE" )
        pe_emm_df = pe_main_df.filter( F.col("CURRENCY_CODE") == "EMM" ).select("SCENARIO_NAME", "SCENARIO_TYPE", "NORMALIZED_BPS_SHOCK").distinct()
        
        pe_main_df = (
            pe_main_df.alias("pe")
            .join(
                pe_emm_df.alias("emm"),
                on = [
                    F.col("pe.SCENARIO_NAME") == F.col("emm.SCENARIO_NAME"),
                    F.col("pe.SCENARIO_TYPE") == F.col("emm.SCENARIO_TYPE")
                ],
                how = "left"
            )
            .select("pe.*", F.col("emm.NORMALIZED_BPS_SHOCK").alias("EMM_NORMALIZED_BPS_SHOCK"))
            .withColumn(
                "SHOCK_FACTOR",
                F.when(
                    F.col("CURRENCY_CODE") == "VND", F.col("NORMALIZED_BPS_SHOCK") + F.col("EMM_NORMALIZED_BPS_SHOCK")
                )
                .otherwise(
                    F.col("NORMALIZED_BPS_SHOCK")
                )
            )
            .drop("EMM_NORMALIZED_BPS_SHOCK")
        )

        unified_shocks_df = (
            unified_shocks_base_df
            .filter( F.col("RISK_FACTOR_SUBTYPE") != "PE" )
            .union(pe_main_df)
        )

        unified_shocks_df = self.cache_df(unified_shocks_df, "unified_shocks_df")

        self.write(unified_shocks_df, self._step_config.step_unified_shocks.table_name, self._step_config.step_unified_shocks.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info(f"Unified shock generation completed")

        return unified_shocks_df
    
    def calculate_consolidated_shocks(self, inputs, unified_shocks_df):

        self._logger.info(f"Calculating consolidated shocks...")
        
        consolidated_shocks_df = (
            unified_shocks_df
            .groupBy("RISK_FACTOR_SUBTYPE", "CURRENCY_CODE", "TENOR", "ES_SENSITIVITY_NAME")
            .agg( F.sum("SHOCK_FACTOR").alias("CONSOLIDATED_SHOCK_FACTOR") )
        )

        consolidated_shocks_df = self.cache_df(consolidated_shocks_df, "consolidated_shocks_df")

        self.write(consolidated_shocks_df, self._step_config.step_consolidated_shocks.table_name, self._step_config.step_consolidated_shocks.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self._logger.info(f"Consolidated shock generation completed")

        return consolidated_shocks_df
