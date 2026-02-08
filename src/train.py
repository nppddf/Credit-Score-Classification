import logging
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from pathlib import Path
import joblib

from config import load_config


CONFIG = load_config()
TRAIN_CONFIG = CONFIG.get("train", {})

RANDOM_STATE = TRAIN_CONFIG["random_state"]
NUMBER_OF_FOLDS = TRAIN_CONFIG["number_of_folds"]

CATEGORICAL = TRAIN_CONFIG["categorical"]
DROP = TRAIN_CONFIG["drop"]

logger = logging.getLogger(__name__)


def drop_columns(dataframe):
    dataframe.drop(DROP, axis=1, inplace=True)


def prepare_xy(dataframe):
    drop_columns(dataframe)

    y_enc = LabelEncoder()
    logger.info("Enconding the data.")
    y = y_enc.fit_transform(dataframe["Credit_Score"])
    logger.info("Enconding completed.")

    X = dataframe.drop(columns=["Credit_Score"])
    return X, y, y_enc


def createPreprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ],
        remainder="passthrough",
    )

    return preprocessor


def createModelPipeline(preprocessor):
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    return model


def createScoring():
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
    }

    return scoring


def validate_model(model, X, y, cv, scoring):
    logger.info("Validating the model.")
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    logger.info("Validation completed.")

    return scores


def printMetrics(scores):
    logger.info("-----------------------")
    logger.info("Average Accuracy: %.4f", scores["test_accuracy"].mean())
    logger.info("Average Precision: %.4f", scores["test_precision"].mean())
    logger.info("Average Recall: %.4f", scores["test_recall"].mean())
    logger.info("-----------------------")


def fitModel(model, X, y):
    logger.info("Fitting the model.")
    model.fit(X, y)
    logger.info("Fitting completed.")


def savePipeline(model):
    model_path = project_root / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Model saved.")


def fit_random_forest(X, y, project_root):
    preprocessor = createPreprocessor()
    model = createModelPipeline(preprocessor)
    scoring = createScoring()

    scores = validate_model(model, X, y, NUMBER_OF_FOLDS, scoring)
    printMetrics(scores)

    fitModel(model, X, y)

    savePipeline(model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_path = project_root / "data" / "processed" / "train_processed.csv"

    train_data = pd.read_csv(data_path)
    X, y, y_enc = prepare_xy(train_data)

    fit_random_forest(X, y, project_root)
