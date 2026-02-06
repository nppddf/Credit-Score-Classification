from numpy import float64, int64


STEP_CONFIGS = [
    {
        "clean_func": "clean_numerical_field",
        "groupby": "Age",
        "column": "Age",
        "strip": "_",
        "datatype": int64,
        "min_value": 0,
        "max_value": 100,
    },
    {
        "clean_func": "clean_categorical_field",
        "groupby": "SSN",
        "column": "SSN",
        "replace_value": "#F%$D@*&8",
    },
    {
        "clean_func": "clean_categorical_field",
        "groupby": "Occupation",
        "column": "Occupation",
        "replace_value": "_______",
    },
    {
        "clean_func": "clean_categorical_field",
        "groupby": "Changed_Credit_Limit",
        "column": "Changed_Credit_Limit",
        "replace_value": "_",
    },
    {
        "clean_func": "clean_categorical_field",
        "groupby": "Credit_Mix",
        "column": "Credit_Mix",
        "replace_value": "_",
    },
    {
        "clean_func": "clean_categorical_field",
        "groupby": "Payment_Behaviour",
        "column": "Payment_Behaviour",
        "replace_value": "!@9#%8",
    },
    {
        "clean_func": "clean_numerical_field",
        "groupby": "Monthly_Balance",
        "column": "Monthly_Balance",
        "replace_value": ",__-333333333333333333333333333__",
        "datatype": float64,
    },
]

TYPICAL_NUMERIC_COLUMNS = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Changed_Credit_Limit",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
]


def add_typical_numeric_columns_to_config():
    for column_name in TYPICAL_NUMERIC_COLUMNS:
        STEP_CONFIGS.append(
            {
                "clean_func": "clean_numerical_field",
                "groupby": column_name,
                "column": column_name,
                "strip": "_",
                "datatype": float64,
                "min_value": 0,
            },
        )


PREPROCESSING_STEPS = []


def preprocessing_step(func):
    PREPROCESSING_STEPS.append(func)
    return func


def register_step(clean_func, groupby, column, **kwargs):
    @preprocessing_step
    def _step(df):
        clean_func(df, groupby, column, **kwargs)
