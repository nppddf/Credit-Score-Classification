import logging
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing_config import (
    STEP_CONFIGS,
    PREPROCESSING_STEPS,
    add_typical_numeric_columns_to_config,
    register_step,
)

logger = logging.getLogger(__name__)


def describe_column(dataframe, column):
    logger.info("Details of %s column", column)
    logger.info("\nDataType: %s", dataframe[column].dtype)

    count_null = dataframe[column].isnull().sum()
    if count_null == 0:
        logger.info("\nThere are no null values")
    elif count_null > 0:
        logger.info("\nThere are %s null values", count_null)

    logger.info("\nNumber of Unique Values: %s", dataframe[column].nunique())
    logger.info("\nDistribution of column:\n%s", dataframe[column].value_counts())


def clean_categorical_field(dataframe, column, replace_value=None):
    if replace_value is not None:
        dataframe[column] = dataframe[column].replace(replace_value, np.nan)
        logger.info("\nGarbage value %s is replaced with np.nan", replace_value)

    fill_missing_with_mode(dataframe, column)


def fill_missing_with_mode(dataframe, column):
    logger.info(
        "\nNo. of missing values before filling with mode: %s",
        dataframe[column].isnull().sum(),
    )

    mode_series = dataframe[column].mode(dropna=True)
    fill_value = mode_series.iat[0] if not mode_series.empty else np.nan
    dataframe[column] = dataframe[column].fillna(fill_value)

    logger.info(
        "\nNo. of missing values after filling with mode: %s",
        dataframe[column].isnull().sum(),
    )


def _apply_dtype(series, datatype):
    if datatype is None:
        return series
    if datatype == "Int64":
        return series.round().astype(datatype)
    return series.astype(datatype)


def clean_numerical_field(
    dataframe,
    column,
    strip=None,
    datatype=None,
    replace_value=None,
    min_value=None,
    max_value=None,
    quantile_lower=None,
    quantile_upper=None,
):
    if replace_value is not None:
        dataframe[column] = dataframe[column].replace(replace_value, np.nan)
        logger.info("Garbage value %s is replaced with np.nan", replace_value)

    is_string_col = pd.api.types.is_string_dtype(dataframe[column])
    if is_string_col and strip is not None:
        dataframe[column] = dataframe[column].astype("string").str.strip(strip)
        logger.info("Trailing & leading %s are removed", strip)

    if datatype is not None or is_string_col:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    fix_inconsistent_values(
        dataframe,
        column,
        min_value=min_value,
        max_value=max_value,
        quantile_lower=quantile_lower,
        quantile_upper=quantile_upper,
    )

    if datatype is not None:
        dataframe[column] = _apply_dtype(dataframe[column], datatype)
        logger.info("Datatype of %s is changed to %s", column, datatype)
    elif is_string_col:
        logger.info("Column %s converted to numeric type", column)


def fix_inconsistent_values(
    dataframe,
    column,
    min_value=None,
    max_value=None,
    quantile_lower=None,
    quantile_upper=None,
):
    series = dataframe[column]
    logger.info(
        "\nExisting Min, Max Values:\n%s",
        series.agg(["min", "max"]),
    )

    lower_bound = None
    upper_bound = None

    if quantile_lower is not None:
        lower_bound = series.quantile(quantile_lower)
        if pd.isna(lower_bound):
            lower_bound = None
    if quantile_upper is not None:
        upper_bound = series.quantile(quantile_upper)
        if pd.isna(upper_bound):
            upper_bound = None

    if min_value is not None:
        lower_bound = min_value if lower_bound is None else max(min_value, lower_bound)
    if max_value is not None:
        upper_bound = max_value if upper_bound is None else min(max_value, upper_bound)

    if lower_bound is not None or upper_bound is not None:
        dataframe[column] = series.clip(lower=lower_bound, upper=upper_bound)
        logger.info(
            "\nClipped %s with bounds: %s, %s",
            column,
            lower_bound,
            upper_bound,
        )

    dataframe[column] = dataframe[column].fillna(dataframe[column].median())

    logger.info(
        "\nAfter Cleaning Min, Max Values:\n%s",
        dataframe[column].agg(["min", "max"]),
    )
    logger.info(
        "\nNo. of Unique values after Cleaning: %s", dataframe[column].nunique()
    )
    logger.info(
        "\nNo. of Null values after Cleaning: %s", dataframe[column].isnull().sum()
    )


def convert_age_to_months(x):
    if pd.isna(x):
        return 0

    parts = x.split()
    years = int(parts[0])
    months = int(parts[3])

    return years * 12 + months


_FUNC_REGISTRY = {
    "clean_numerical_field": clean_numerical_field,
    "clean_categorical_field": clean_categorical_field,
}

add_typical_numeric_columns_to_config()

for cfg in STEP_CONFIGS:
    func_name = cfg["clean_func"]
    clean_func = _FUNC_REGISTRY[func_name]
    other_args = {k: v for k, v in cfg.items() if k != "clean_func"}
    register_step(clean_func, **other_args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_path = project_root / "data" / "raw" / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please run 'python src/download_dataset.py' first to download the dataset."
        )

    RAW_DATA = pd.read_csv(data_path, low_memory=False)
    TRAIN_DATA = RAW_DATA.copy()

    destination = project_root / "data" / "processed" / "train_processed.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    for step in PREPROCESSING_STEPS:
        step(TRAIN_DATA)

    TRAIN_DATA["Credit_History_Age"] = TRAIN_DATA["Credit_History_Age"].apply(
        convert_age_to_months
    )
    logger.info("\nColumn 'Credit_History_Age' has cleaned\n")

    TRAIN_DATA.to_csv(destination, index=False)
