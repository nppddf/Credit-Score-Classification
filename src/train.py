import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
import joblib

RANDOM_STATE = 42
TEST_SIZE = 0.2


def drop_columns(dataframe):
    columns_to_drop = [
        "ID",
        "Customer_ID",
        "Month",
        "Name",
        "SSN",
    ]

    dataframe.drop(columns_to_drop, axis=1, inplace=True)


def encoding_labels(dataframe):
    categorical_columns = [
        "Occupation",
        "Type_of_Loan",
        "Credit_Mix",
        "Payment_of_Min_Amount",
        "Payment_Behaviour",
        "Credit_Score",
    ]

    label_encoder = LabelEncoder()
    for column in categorical_columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])


def fit_random_forest(X_train, y_train):
    random_forest = RandomForestClassifier()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
    }

    scores = cross_validate(
        random_forest, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1
    )

    avg_accuracy = scores["test_accuracy"].mean()
    avg_precision = scores["test_precision"].mean()
    avg_recall = scores["test_recall"].mean()

    print("-----------------------")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print("-----------------------")

    joblib.dump(random_forest, "../models/model.joblib")


if __name__ == "__main__":
    train_data = pd.read_csv("../data/processed/train_processed.csv")

    drop_columns(train_data)
    encoding_labels(train_data)

    X = train_data.drop("Credit_Score", axis=1)
    y = train_data["Credit_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    fit_random_forest(X_train, y_train)
