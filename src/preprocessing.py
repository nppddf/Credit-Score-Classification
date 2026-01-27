import numpy as np
import statistics as stats
import pandas as pd
from pathlib import Path


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
    modes = df_dropped.apply(lambda x: stats.mode(x))
    mini, maxi = modes.min(), modes.max()

    col = df[column].apply(
        lambda x: np.nan if ((x < mini) | (x > maxi) | (x < 0)) else x
    )

    mode_by_group = df.groupby(groupby)[column].transform(
        lambda x: x.median() if len(x) > 0 else np.nan
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
        df[column] = pd.to_numeric(df[column], errors='coerce')
        print(f"\nDatatype of {column} is changed to {datatype}")
    elif df[column].dtype == object:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        print(f"\nColumn {column} converted to numeric type")

    fix_inconsistent_values(df, groupby, column)

PREPROCESSING_STEPS = []

def preprocessing_step(func):
    PREPROCESSING_STEPS.append(func)
    return func


def register_step(clean_func, groupby, column, **kwargs):
    @preprocessing_step
    def _step(df):
        clean_func(df, groupby, column, **kwargs)

    return _step


register_step(
    clean_numerical_field,
    "Monthly_Balance",
    "Monthly_Balance",
    replace_value=",__-333333333333333333333333333__",
    datatype=float,
)

register_step(
    clean_categorical_field,
    "SSN",
    "SSN",
    replace_value="#F%$D@*&8",
)

register_step(
    clean_categorical_field,
    "Occupation",
    "Occupation",
    replace_value="_______",
)

register_step(
    clean_categorical_field,
    "Amount_invested_monthly",
    "Amount_invested_monthly",
    replace_value="__10000__",
)

register_step(
    clean_categorical_field,
    "Changed_Credit_Limit",
    "Changed_Credit_Limit",
    replace_value="_",
)

register_step(
    clean_categorical_field,
    "Credit_Mix",
    "Credit_Mix",
    replace_value="_",
)

register_step(
    clean_numerical_field,
    "Num_of_Loan",
    "Num_of_Loan",
    strip="_",
    datatype=float,
)

register_step(
    clean_categorical_field,
    "Payment_Behaviour",
    "Payment_Behaviour",
    replace_value="!@9#%8",
)

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
    
    destination = project_root / "data" / "new" / "train_backup.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    #for col in TRAIN_DATA.columns:
    #    describe_column(RAW_DATA, col)
    for step in PREPROCESSING_STEPS:
        step(TRAIN_DATA)

    TRAIN_DATA.to_csv(destination, index=False)    
    #describe_column(TRAIN_DATA,"Occupation")
    # for col in TRAIN_DATA.columns:
    #     if col.dtype in ("int64", "float64"):
    #         clean_numerical_field(TRAIN_DATA, ...)
    #     else:
    #         clean_categorical_field(TRAIN_DATA, ...)
