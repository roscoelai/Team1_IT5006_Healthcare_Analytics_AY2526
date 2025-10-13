"""Custom model pipeline components for the diabetes readmission dataset."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to apply log transformation to specified columns."""

    def __init__(self, columns: list[str]):
        self.columns: list[str] = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "LogTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col] = np.log1p(X_transformed[col])
        return X_transformed


def stratified_split(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train-test split.

    Args:
        data (pd.DataFrame): The input dataframe.
        features (list[str]): List of feature column names.
        target (str): The target column name.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def get_gaussian_nb_pipeline() -> Pipeline:
    """Get a pipeline with preprocessing and Gaussian Naive Bayes classifier.

    Returns:
        Pipeline: A sklearn Pipeline object with preprocessing and GaussianNB classifier.
    """

    # Define the column transformer
    column_transform = ColumnTransformer(
        transformers=[
            # OHE for diag_1, diag_2, diag_3
            (
                "ohe",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                ["diag_1", "diag_2", "diag_3"],
            ),
            (
                "race_ordinal",
                OrdinalEncoder(
                    categories=[
                        [
                            "Caucasian",
                            "AfricanAmerican",
                            "Hispanic",
                            "Asian",
                            "Other",
                        ],
                        ["Male", "Female"],
                        ["None", "Norm", ">200", ">300"],
                        ["None", "Norm", ">7", ">8"],
                        ["No", "Steady", "Change"],
                        ["No", "Ch"],
                        ["No", "Yes"],
                        ["No", "Steady", "Change"],
                        ["No", "Steady", "Change"],
                        ["No", "Steady", "Change"],
                        ["No", "Steady", "Change"],
                        ["No", "Steady", "Change"],
                        ["No", "Steady", "Change"],
                        [
                            "[0-10)",
                            "[10-20)",
                            "[20-30)",
                            "[30-40)",
                            "[40-50)",
                            "[50-60)",
                            "[60-70)",
                            "[70-80)",
                            "[80-90)",
                            "[90-100)",
                        ],
                    ]
                ),
                [
                    "race",
                    "gender",
                    "max_glu_serum",
                    "A1Cresult",
                    "insulin",
                    "change",
                    "diabetesMed",
                    "metformin",
                    "glimepiride",
                    "glipizide",
                    "glyburide",
                    "pioglitazone",
                    "rosiglitazone",
                    "age",
                ],
            ),
            (
                "log_transform",
                LogTransformer(
                    columns=[
                        "time_in_hospital",
                        "num_lab_procedures",
                        "num_procedures",
                        "num_medications",
                        "number_outpatient",
                    ],
                ),
                [
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_outpatient",
                ],
            ),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("column_transform", column_transform),
            ("classifier", GaussianNB()),
        ],
    )

    return pipe


def train_gaussian_nb(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[GaussianNB, pd.Series, np.ndarray, np.ndarray]:
    """Train a Gaussian Naive Bayes model with preprocessing pipeline.

    Args:
        data (pd.DataFrame): The input dataframe.
        features (list[str]): List of feature column names.
        target (str): The target column name.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        tuple[GaussianNB, pd.Series, np.ndarray, np.ndarray]: Trained GaussianNB model,
            true labels, predicted labels, and predicted probabilities for the test set.

    """

    X_train, X_test, y_train, y_test = stratified_split(
        data=data,
        features=features,
        target=target,
        test_size=test_size,
        random_state=random_state,
    )

    pipe = get_gaussian_nb_pipeline()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)
    model = pipe.named_steps["classifier"]

    return model, y_test, y_pred, y_proba


def evaluate_model(y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray):
    """Evaluate the model performance with plots and summary table."""

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix as percentages
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", ax=ax[0])
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    ax[0].set_title("Confusion Matrix (%)")

    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax[1].plot([0, 1], [0, 1], "k--")
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].set_title("ROC Curve")
        ax[1].legend()
    else:
        ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Metrics summary table
    metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
            "Score": [accuracy, precision, recall, f1, roc_auc],
        }
    )
    print("\nModel Performance Summary:")
    print(metrics)
