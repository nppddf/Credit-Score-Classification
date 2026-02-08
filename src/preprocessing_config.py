from config import load_config


_CONFIG = load_config()
_PREPROCESSING_CONFIG = _CONFIG.get("preprocessing", {})

_DTYPE_MAP = {
    "float64": "float64",
    "int64": "Int64",
}

_NUMERIC_CLIP = _PREPROCESSING_CONFIG.get("numeric_clip_quantiles", {})
_DEFAULT_QUANTILE_LOWER = _NUMERIC_CLIP.get("lower")
_DEFAULT_QUANTILE_UPPER = _NUMERIC_CLIP.get("upper")


def _normalize_dtype(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value in _DTYPE_MAP:
            return _DTYPE_MAP[value]
        raise ValueError(f"Unsupported datatype '{value}' in preprocessing config")
    return value


def _apply_default_quantiles(cfg):
    if cfg.get("clean_func") != "clean_numerical_field":
        return cfg
    if _DEFAULT_QUANTILE_LOWER is not None and "quantile_lower" not in cfg:
        cfg["quantile_lower"] = _DEFAULT_QUANTILE_LOWER
    if _DEFAULT_QUANTILE_UPPER is not None and "quantile_upper" not in cfg:
        cfg["quantile_upper"] = _DEFAULT_QUANTILE_UPPER
    return cfg


STEP_CONFIGS = []
for step in _PREPROCESSING_CONFIG.get("step_configs", []):
    cfg = dict(step)
    if "datatype" in cfg:
        cfg["datatype"] = _normalize_dtype(cfg["datatype"])
    cfg = _apply_default_quantiles(cfg)
    STEP_CONFIGS.append(cfg)

TYPICAL_NUMERIC_COLUMNS = _PREPROCESSING_CONFIG.get("typical_numeric_columns", [])


def add_typical_numeric_columns_to_config():
    for column_name in TYPICAL_NUMERIC_COLUMNS:
        step_cfg = {
            "clean_func": "clean_numerical_field",
            "column": column_name,
            "strip": "_",
            "datatype": _DTYPE_MAP["float64"],
            "min_value": 0,
        }
        if _DEFAULT_QUANTILE_LOWER is not None:
            step_cfg["quantile_lower"] = _DEFAULT_QUANTILE_LOWER
        if _DEFAULT_QUANTILE_UPPER is not None:
            step_cfg["quantile_upper"] = _DEFAULT_QUANTILE_UPPER
        STEP_CONFIGS.append(step_cfg)


PREPROCESSING_STEPS = []


def preprocessing_step(func):
    PREPROCESSING_STEPS.append(func)
    return func


def register_step(clean_func, column, **kwargs):
    @preprocessing_step
    def _step(df):
        clean_func(df, column, **kwargs)
