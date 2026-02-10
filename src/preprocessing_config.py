from copy import deepcopy

from config import load_config

_DTYPE_MAP = {
    "float64": "float64",
    "int64": "Int64",
    "Int64": "Int64",
}


def normalize_dtype(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value in _DTYPE_MAP:
            return _DTYPE_MAP[value]
        raise ValueError(f"Unsupported datatype '{value}' in preprocessing config")
    return value


def load_preprocessing_columns():
    config = load_config()
    preprocessing = config.get("preprocessing", {})
    columns = preprocessing.get("columns", {})
    return deepcopy(columns)
