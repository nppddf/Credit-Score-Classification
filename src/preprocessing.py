import numpy as np
import statistics as stats
import pandas as pd


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


def clean_categorical_field(df, groupby, column, replace_value=None):
    if replace_value != None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"\nGarbage value {replace_value} is replaced with np.nan")

    fill_missing_with_group_mode(df, groupby, column)


def fix_inconsistent_values(df, groupby, column):
    print(
        "\nExisting Min, Max Values:", df[column].apply([min, max]), sep="\n", end="\n"
    )

    df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
    x, y = df_dropped.apply(lambda x: stats.mode(x)).apply([min, max])
    mini, maxi = x[0][0], y[0][0]

    col = df[column].apply(
        lambda x: np.nan if ((x < mini) | (x > maxi) | (x < 0)) else x
    )

    mode_by_group = df.groupby(groupby)[column].transform(
        lambda x: x.median()[0] if not x.mode().empty else np.nan
    )
    df[column] = col.fillna(mode_by_group)
    df[column].fillna(df[column].median(), inplace=True)

    print(
        "\nAfter Cleaning Min, Max Values:",
        df[column].apply([min, max]),
        sep="\n",
        end="\n",
    )
    print("\nNo. of Unique values after Cleaning:", df[column].nunique())
    print("\nNo. of Null values after Cleaning:", df[column].isnull().sum())


def clean_numerical_field(
    df, groupby, column, strip=None, datatype=None, replace_value=None
):
    if replace_value != None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"\nGarbage value {replace_value} is replaced with np.nan")

    if df[column].dtype == object and strip is not None:
        df[column] = df[column].str.strip(strip)
        print(f"\nTrailing & leading {strip} are removed")

    if datatype is not None:
        df[column] = df[column].astype(datatype)
        print(f"\nDatatype of {column} is changed to {datatype}")

    fix_inconsistent_values(df, groupby, column)


if __name__ == "__main__":
    TRAIN_DATA = pd.read_csv("data/raw/train.csv")

    for col in TRAIN_DATA.columns:
        describe_column(TRAIN_DATA, col)

    # for col in TRAIN_DATA.columns:
    #     if col.dtype in ("int64", "float64"):
    #         clean_numerical_field(TRAIN_DATA, ...)
    #     else:
    #         clean_categorical_field(TRAIN_DATA, ...)
