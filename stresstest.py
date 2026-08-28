from pyspark.sql import DataFrame
from pyspark.sql import functions as F, Window
from typing import NamedTuple
from rda.engine.base_step import RDABaseStep
import rda.utils.spark_utils as su

class StressTestInputs(NamedTuple):
    portfolio_details_df: DataFrame
    ga_b1_df: DataFrame


class StressTest(RDABaseStep):

    def read(self) -> StressTestInputs:
        inputs = self._step_config.inputs
        self._logger.info(f"Reading inputs for Stress test...")

        input_dfs = StressTestInputs(
            portfolio_details_df = self._uc.read_latest(
                inputs.portfolio_details.table_name,
                watermark_col=inputs.portfolio_details.watermark_col
            ),
            ga_b1_df = self._uc.read_latest(
                inputs.ga_b1.table_name,
                filter=f"SCENARIO_NAME in ('2008 - 2009 Global Financial Crisis - Credit extended', '2008 - 2009 Global Financial Crisis - Credit extended - No Curr') AND RISK_MODEL = 'MAC.L'",
                watermark_col=inputs.ga_b1.watermark_col
            )
        )

        self.validate_inputs(input_dfs)
        return input_dfs
    

    def step_asset_impact(self, inputs: StressTestInputs) -> DataFrame:
        self._logger.info(f"Calculating asset impact...")
        
        portfolio_details_subset_df = (
            inputs.portfolio_details_df
            .select("PORTFOLIO_CODE", "TAM_SEGMENT", "LEGAL_ID", "SUBFUND")
            .distinct()
        )

        ga_b1_base_df = (
            inputs.ga_b1_df.filter(F.col("SCENARIO_NAME") == "2008 - 2009 Global Financial Crisis - Credit extended").alias("a")
            .join(
                inputs.ga_b1_df.filter(F.col("SCENARIO_NAME") == "2008 - 2009 Global Financial Crisis - Credit extended - No Curr").alias("b"),
                on = "PORTFOLIO_CODE"
            )
            .select(
                F.col("a.*"),
                F.col("b.PNL_AMT").alias("PNL_AMT_NO_FX")
            )
        )

        ga_b1_enhanced_df = (
            ga_b1_base_df.alias("a")
            .join(
                portfolio_details_subset_df.alias("b"),
                on = "PORTFOLIO_CODE"
            )
            .select("a.*", "b.TAM_SEGMENT", "b.LEGAL_ID", "b.SUBFUND")
        )

        ga_b1_derive_df = (
            ga_b1_enhanced_df
            .withColumn(
                "ES_SEGMENT_NAME",
                F.when( F.col("TAM_SEGMENT") == "JH USD-Retail and Institutional UL", "JH USD-Retail and Institutional UL" )
                .when( F.col("TAM_SEGMENT") == "JH USD-Guaranteed", "JH USD-Guaranteed" )
                .when( F.col("TAM_SEGMENT") == "JHUSA USD-Surplus", "US SURPLUS" )
                .when( ((F.col("LEGAL_ID") == "19") & (F.col("SUBFUND") == "145")), "VUL Retail/Inst'l" )
                .when( ((F.col("LEGAL_ID") == "102") & (F.col("SUBFUND") == "146")), "US Surplus" )
                .when( F.col("PORTFOLIO_CODE") == "0019_099_COLI", "External COLI" )
                .otherwise(F.col("TAM_SEGMENT"))
            )
            .withColumn("FX_IMPACT", F.col("PNL_AMT") - F.col("PNL_AMT_NO_FX"))
            .withColumn("PCNT_MV_CHANGE_NO_FX", F.col("PNL_AMT_NO_FX") * 100 / F.col("INITIAL_MARKET_VALUE" ))
            .select(
                "PORTFOLIO_CODE",
                "ES_SEGMENT_NAME",
                "ADJUSTED_ASSET_GROUP",
                "ANALYSIS_DATE",
                "SCENARIO_NAME",
                "RISK_MODEL",
                "INITIAL_MARKET_VALUE",
                F.col("PNL_AMT").alias("MV_CHANGE_WITH_FX"),
                F.col("PNL_AMT_NO_FX").alias("MV_CHANGE_NO_FX"),
                "FX_IMPACT",
                "PCNT_MV_CHANGE_NO_FX"
            )
        )

        ga_b1_derive_df = self.cache_df(ga_b1_derive_df, "ga_b1_derive_df")
        ga_b1_derive_alda_pe_df = ga_b1_derive_df.filter( F.col("ADJUSTED_ASSET_GROUP").isin(["ALDA", "Public Equity"]) )

        self.write(ga_b1_derive_df, self._step_config.step_asset_impact_ga.table_name, self._step_config.step_asset_impact_ga.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        self.write(ga_b1_derive_alda_pe_df, self._step_config.step_asset_impact_pe_alda.table_name, self._step_config.step_asset_impact_pe_alda.ext_table_loc, mode=RDABaseStep.spark_write_mode)
        
        self._logger.info("Asset impact calculation completed.")
        return (ga_b1_derive_df, ga_b1_derive_alda_pe_df)
    