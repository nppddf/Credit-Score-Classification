import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing_config import load_preprocessing_columns, normalize_dtype

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


def replace_values(dataframe, column, value):
    if value is None:
        return
    dataframe[column] = dataframe[column].replace(value, np.nan)
    logger.info("Garbage value %s is replaced with np.nan", value)


def strip_strings(dataframe, column, chars):
    if not pd.api.types.is_string_dtype(dataframe[column]):
        return
    dataframe[column] = dataframe[column].astype("string").str.strip(chars)
    logger.info("Trailing & leading %s are removed", chars)


def to_numeric(dataframe, column):
    dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    logger.info("Column %s converted to numeric type", column)


def _fill_missing(dataframe, column, label, value_getter, invalid_mask=None):
    logger.info(
        "\nNo. of missing values before filling with %s: %s",
        label,
        dataframe[column].isnull().sum(),
    )

    series = dataframe[column]
    missing_mask = series.isna()
    if invalid_mask is None:
        fill_mask = missing_mask
    else:
        fill_mask = missing_mask & ~invalid_mask

    fill_value = value_getter(series)
    if fill_mask.any():
        dataframe[column] = series.mask(fill_mask, fill_value)

    logger.info(
        "\nNo. of missing values after filling with %s: %s",
        label,
        dataframe[column].isnull().sum(),
    )


def _mode_value(series):
    mode_series = series.mode(dropna=True)
    return mode_series.iat[0] if not mode_series.empty else np.nan


def fill_missing_with_mode(dataframe, column, invalid_mask=None):
    _fill_missing(dataframe, column, "mode", _mode_value, invalid_mask=invalid_mask)


def fill_missing_with_median(dataframe, column, invalid_mask=None):
    _fill_missing(
        dataframe,
        column,
        "median",
        lambda series: series.median(),
        invalid_mask=invalid_mask,
    )


def _compute_bounds(
    series,
    min_value=None,
    max_value=None,
    quantile_lower=None,
    quantile_upper=None,
):
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

    return lower_bound, upper_bound


def apply_bounds(
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

    lower_bound, upper_bound = _compute_bounds(
        series,
        min_value=min_value,
        max_value=max_value,
        quantile_lower=quantile_lower,
        quantile_upper=quantile_upper,
    )

    if lower_bound is None and upper_bound is None:
        return

    mask = pd.Series(True, index=series.index)
    if lower_bound is not None:
        mask &= series >= lower_bound
    if upper_bound is not None:
        mask &= series <= upper_bound

    dataframe[column] = series.where(mask, np.nan)
    logger.info(
        "\nMasked values for %s with bounds: %s, %s",
        column,
        lower_bound,
        upper_bound,
    )


def _apply_dtype(series, datatype):
    if datatype is None:
        return series
    if datatype == "Int64":
        return series.round().astype(datatype)
    return series.astype(datatype)


def cast_dtype(dataframe, column, dtype):
    dataframe[column] = _apply_dtype(dataframe[column], dtype)
    logger.info("Datatype of %s is changed to %s", column, dtype)


def convert_age_to_months(x):
    if pd.isna(x):
        return np.nan

    parts = x.split()
    years = int(parts[0])
    months = int(parts[3])

    return years * 12 + months


def convert_age_column_to_months(dataframe, column):
    dataframe[column] = dataframe[column].apply(convert_age_to_months)
    logger.info("Column %s converted to months", column)


def _invalid_mask_for_replace(series, value):
    if value is None:
        return pd.Series(False, index=series.index)
    if isinstance(value, dict):
        return series.isin(value.keys())
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return series.isin(value)
    return series == value


_OP_REGISTRY = {
    "replace": replace_values,
    "strip": strip_strings,
    "to_numeric": to_numeric,
    "bounds": apply_bounds,
    "fill_mode": fill_missing_with_mode,
    "fill_median": fill_missing_with_median,
    "cast": cast_dtype,
    "convert_age_to_months": convert_age_column_to_months,
}


def _normalize_operation(operation):
    if isinstance(operation, str):
        return operation, {}
    if not isinstance(operation, dict):
        raise ValueError(f"Unsupported operation format: {operation}")

    op_name = operation.get("op")
    if op_name is None:
        raise ValueError(f"Missing 'op' in operation: {operation}")

    kwargs = {key: value for key, value in operation.items() if key != "op"}
    if "dtype" in kwargs:
        kwargs["dtype"] = normalize_dtype(kwargs["dtype"])

    return op_name, kwargs


def apply_operations(dataframe, column, operations):
    invalid_mask = pd.Series(False, index=dataframe.index)

    for operation in operations:
        op_name, kwargs = _normalize_operation(operation)
        func = _OP_REGISTRY.get(op_name)
        if func is None:
            raise ValueError(f"Unsupported operation '{op_name}'")

        if op_name == "replace":
            invalid_mask |= _invalid_mask_for_replace(
                dataframe[column],
                kwargs.get("value"),
            )
            func(dataframe, column, **kwargs)
            continue

        if op_name == "to_numeric":
            before_na = dataframe[column].isna()
            func(dataframe, column, **kwargs)
            after_na = dataframe[column].isna()
            invalid_mask |= (~before_na) & after_na
            continue

        if op_name in {"fill_mode", "fill_median"}:
            kwargs = dict(kwargs)
            kwargs["invalid_mask"] = invalid_mask
            func(dataframe, column, **kwargs)
            continue

        func(dataframe, column, **kwargs)


def preprocess_dataframe(dataframe, columns_config):
    for column, operations in columns_config.items():
        apply_operations(dataframe, column, operations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    columns_config = load_preprocessing_columns()
    if not columns_config:
        raise ValueError("No preprocessing columns configured.")

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_path = project_root / "data" / "raw" / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Please run 'python src/download_dataset.py' first to download the dataset."
        )

    raw_data = pd.read_csv(data_path, low_memory=False)
    train_data = raw_data.copy()

    destination = project_root / "data" / "processed" / "train_processed.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    preprocess_dataframe(train_data, columns_config)

    train_data.to_csv(destination, index=False)
    logger.info("Processed data saved to %s", destination)
