import numpy as np
import statistics as stats
import pandas as pd
from pathlib import Path

from preprocessing_config import (
    STEP_CONFIGS,
    PREPROCESSING_STEPS,
    add_typical_numeric_columns_to_config,
    register_step,
)


def describe_column(df, column):
    print("Details of", column, "column")

    print("\nDataType: ", df[column].dtype)

    count_null = df[column].isnull().sum()
    if count_null == 0:
        print("\nThere are no null values")
    elif count_null > 0:
        print("\nThere are ", count_null, " null values")

    print("\nNumber of Unique Values: ", df[column].nunique())

    print("\nDistribution of column:\n")
    print(df[column].value_counts())


def clean_categorical_field(df, groupby, column, replace_value=None):
    if replace_value is not None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"\nGarbage value {replace_value} is replaced with np.nan")

    fill_missing_with_group_mode(df, groupby, column)


def fill_missing_with_group_mode(df, groupby, column):
    print(
        "\nNo. of missing values before filling with group mode:",
        df[column].isnull().sum(),
    )

    mode_per_group = df.groupby(groupby)[column].transform(lambda x: x.mode().iat[0])
    df[column] = df[column].fillna(mode_per_group)

    print(
        "\nNo. of missing values after filling with group mode:",
        df[column].isnull().sum(),
    )


def clean_numerical_field(
    df,
    groupby,
    column,
    strip=None,
    datatype=None,
    replace_value=None,
    min_value=None,
    max_value=None,
):
    if replace_value is not None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"Garbage value {replace_value} is replaced with np.nan")

    is_string_col = pd.api.types.is_string_dtype(df[column])
    if is_string_col and strip is not None:
        df[column] = df[column].astype("string").str.strip(strip)
        print(f"Trailing & leading {strip} are removed")

    if datatype is not None:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        print(f"Datatype of {column} is changed to {datatype}")
    elif is_string_col:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        print(f"Column {column} converted to numeric type")

    fix_inconsistent_values(
        df,
        groupby,
        column,
        min_value=min_value,
        max_value=max_value,
    )


def fix_inconsistent_values(df, groupby, column, min_value=None, max_value=None):
    print(
        "\nExisting Min, Max Values:", df[column].apply([min, max]), sep="\n", end="\n"
    )

    df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
    modes = df_dropped.apply(lambda x: stats.mode(x))
    mini, maxi = modes.min(), modes.max()

    lower_bound = min_value if min_value is not None else mini
    upper_bound = max_value if max_value is not None else maxi

    col = df[column].apply(
        lambda x: np.nan if ((x < lower_bound) | (x > upper_bound) | (x < 0)) else x
    )

    mode_by_group = df.groupby(groupby)[column].transform(
        lambda x: x.mode() if len(x) > 0 else np.nan
    )
    df[column] = col.fillna(mode_by_group)
    df[column] = df[column].fillna(df[column].median())

    print(
        "\nAfter Cleaning Min, Max Values:",
        df[column].apply([min, max]),
        sep="\n",
        end="\n",
    )
    print("\nNo. of Unique values after Cleaning:", df[column].nunique())
    print("\nNo. of Null values after Cleaning:", df[column].isnull().sum())


def convert_age_to_months(x):
    if pd.isna(x):
        return None

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
    print("\nColumn 'Credit_History_Age' has cleaned\n")

    TRAIN_DATA.to_csv(destination, index=False)
