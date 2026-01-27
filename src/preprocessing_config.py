STEP_CONFIGS = [
    {
        "clean_func": "clean_numerical_field",
        "groupby": "Monthly_Balance",
        "column": "Monthly_Balance",
        "replace_value": ",__-333333333333333333333333333__",
        "datatype": float,
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
        "groupby": "Amount_invested_monthly",
        "column": "Amount_invested_monthly",
        "replace_value": "__10000__",
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
        "clean_func": "clean_numerical_field",
        "groupby": "Num_of_Loan",
        "column": "Num_of_Loan",
        "strip": "_",
        "datatype": float,
    },
    {
        "clean_func": "clean_categorical_field",
        "groupby": "Payment_Behaviour",
        "column": "Payment_Behaviour",
        "replace_value": "!@9#%8",
    },
]

