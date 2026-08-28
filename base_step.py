from pyspark.sql import DataFrame, functions as F
from abc import ABC, abstractmethod
from rda.io.unity_catalog import UnityCatalog
from typing import NamedTuple
from datetime import datetime, date
from zoneinfo import ZoneInfo
import logging
from rda.utils.logger import get_logger

class RDABaseStep(ABC):
    spark_write_mode = "append"

    def __init__(self, env_config, spark, step_config, valuation_date=None):
        self._logger = get_logger(self.__class__.__name__)
        self._cached_dfs = dict()
        self._env_config = env_config
        self._spark = spark
        self._step_config = step_config
        self._uc = UnityCatalog(env_config, spark)
        self._date_ctx = self._get_date_ctx()
        self._valuation_date = valuation_date
    
    @abstractmethod
    def read(self):
        raise NotImplementedError(f"Method read() is not implemented for {self.__class__.__name__}")
    
    def read_latest(self, table_name: str): 
        return self._uc.read_latest(table_name)
    
    def _get_date_ctx(self):
        today_key = datetime.now(tz=ZoneInfo("America/Toronto")).strftime("%Y%m%d")
        #today_key = date(2026, 5, 11).strftime("%Y%m%d")
        self._logger.info(f"Today's date key is {today_key}")
        df = self._uc.read_query(
        f"""
            SELECT a.*, b.FORECAST_DATE_KEY, b.FORECAST_DATE, b.FORECAST_SEQ FROM 
            ( select * from {self._env_config.cache.irr_mapping_dim_date.target_table} where INGESTION_TS = ( SELECT MAX(INGESTION_TS) FROM {self._env_config.cache.irr_mapping_dim_date.target_table}) AND DATE_KEY = {today_key} ) a
            join 
            ( select * from {self._env_config.cache.irr_mapping_dim_forecast_date.target_table} where INGESTION_TS = ( SELECT MAX(INGESTION_TS) FROM {self._env_config.cache.irr_mapping_dim_forecast_date.target_table}) AND FORECAST_DATE = (
            select max(FORECAST_DATE) from {self._env_config.cache.irr_mapping_dim_forecast_date.target_table} where FORECAST_DATE_KEY <= {today_key}
            ) 
            ) b
        """
        )
        rows = df.collect()
        if not rows:
            raise ValueError(f"Date key {today_key} not found in {self._env_config.cache.irr_mapping_dim_date.target_table} or table {self._env_config.cache.irr_mapping_dim_forecast_date.target_table} not found...")
        return rows[0]
    
    def add_audit_cols_reporting(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
            .withColumn("REPORTING_DATE", F.lit(self._date_ctx.PREVIOUSQUARTERENDDATE))
            .withColumn("RUN_DATE", F.to_date(F.col("INGESTION_TS")) )
        )
    
    def add_audit_cols_forecasting(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
            .withColumn("REPORTING_DATE", F.lit(self._date_ctx.PREVIOUSQUARTERENDDATE))
            .withColumn("FORECAST_DATE", F.to_date(F.col("INGESTION_TS")) )
            .withColumn("RUN_DATE", F.to_date(F.col("INGESTION_TS")) )
        )
    
    def add_audit_cols_daily_forecasting(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
            .withColumn("REPORTING_DATE", F.lit(self._date_ctx.PREVIOUSQUARTERENDDATE))
            .withColumn("FORECAST_DATE", F.lit(self._valuation_date).cast("date"))
            .withColumn("RUN_DATE", F.to_date(F.col("INGESTION_TS")) )
        )
    
    def add_audit_cols_stresstest(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
            .withColumn("REPORTING_DATE", F.lit(self._valuation_date).cast("date") )
            .withColumn("RUN_DATE", F.to_date(F.col("INGESTION_TS")) )
        )
    
    def add_audit_cols_stresstest(self, df: DataFrame) -> DataFrame:
        return (
            df.withColumn("INGESTION_TS", F.from_utc_timestamp(F.current_timestamp(), 'America/Toronto'))
            .withColumn("RUN_DATE", F.to_date(F.col("INGESTION_TS")) )
        )
    
    def validate_inputs(self, inputs: NamedTuple) -> None:
        empty_dfs = []
        self._logger.info(f"Validating the input dataframes for {self.__class__.__name__}")

        for field in inputs._fields:
            value = getattr(inputs, field)

            if len(value.head(1)) == 0:
                empty_dfs.append(field)                
        
        if len(empty_dfs) > 0:
            raise ValueError(f"The following input dataframes are empty: {", ".join(empty_dfs)}")
    
    def cache_df(self, df: DataFrame, name: str) -> DataFrame:
        self._logger.info(f"Caching dataframe {name} for {self.__class__.__name__}...")
        df.cache()
        self._cached_dfs[name] = df
        return df

    def write(self, df: DataFrame, table_name: str, table_ext_path, mode="append", partition_col: str = None) -> None: 
        if 'ear_report' in table_name:
            self._logger.info("Adding audit columns for EAR reporting")
            df = self.add_audit_cols_reporting(df)
        elif 'ec_forecast' in table_name and 'daily' in table_name:
            self._logger.info("Adding audit columns for EC forecasting")
            df = self.add_audit_cols_daily_forecasting(df)
        elif 'ec_forecast' in table_name:
            self._logger.info("Adding audit columns for EC forecasting")
            df = self.add_audit_cols_forecasting(df)
        elif 'stress' in table_name:
            self._logger.info("Adding audit columns for Stresstest")
            df = self.add_audit_cols_stresstest(df)
        else:
            raise ValueError(f"Table name {table_name} needs implementation for audit columns...")
        self._logger.info(f"Writting data into '{table_name}' table")
        self._uc.write(df, table_name, table_ext_path, mode, partition_col)

    def unpersist_all(self) -> None:
        for name, df in self._cached_dfs.items():
            self._logger.info(f"Unpersisting dataframe {name} for {self.__class__.__name__}...")
            df.unpersist(blocking=False)
