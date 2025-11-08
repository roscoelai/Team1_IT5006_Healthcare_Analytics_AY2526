"""Custom model pipeline components for the diabetes readmission dataset."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
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

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if getattr(self, "feature_names_in_", None) is not None:
                return np.asarray(self.feature_names_in_)
            return np.asarray(self.columns)
        return np.asarray(input_features)


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


def get_gaussian_nb_pipeline(
    use_smote: bool = False, polynomial_degree: int = 2
) -> Pipeline:
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
                "ordinal",
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
            (
                "poly_features",
                PolynomialFeatures(degree=polynomial_degree, include_bias=False),
                [
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                    "number_diagnoses",
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

    if use_smote:
        imb_pipe = ImbPipeline(
            steps=[
                ("column_transform", column_transform),
                ("smote", SMOTE(random_state=42)),
                ("classifier", GaussianNB()),
            ]
        )
        return imb_pipe

    return pipe


def train_gaussian_nb(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Train a Gaussian Naive Bayes model with preprocessing pipeline.

    Args:
        data (pd.DataFrame): The input dataframe.
        features (list[str]): List of feature column names.
        target (str): The target column name.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.3.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
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
    y_pred_test = pipe.predict(X_test)
    y_proba_test = pipe.predict_proba(X_test)
    y_pred_train = pipe.predict(X_train)
    y_proba_train = pipe.predict_proba(X_train)

    return (
        pipe,
        y_test,
        y_pred_test,
        y_proba_test,
        y_train,
        y_pred_train,
        y_proba_train,
    )


def cross_validate_gaussian_nb(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    cv: int = 5,
    scoring: list[str] | str | None = None,
    n_jobs: int = -1,
):
    """Run cross_validate and produce CV train/test summary plus OOF diagnostics.

    Returns:
        tuple: (cv_results, y_oof, y_oof_proba)
    """
    X_train, X_test, y_train, y_test = stratified_split(
        data=data, features=features, target=target, test_size=0.3, random_state=42
    )

    pipe = get_gaussian_nb_pipeline()

    # Helper to compute metrics robustly for binary/multiclass and missing proba
    def compute_metrics(y_true, y_pred, y_proba, class_order: np.ndarray | None = None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall_weighted": recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "roc_auc": None,
        }

        if y_proba is None:
            return metrics

        try:
            classes = np.unique(y_true) if class_order is None else class_order
            n_classes = len(classes)

            if n_classes == 2:
                # Choose positive class: prefer label 1 if present, else the second class
                pos_label = 1 if 1 in classes else classes[1]
                pos_idx = list(classes).index(pos_label)
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, pos_idx])
            else:
                # Multiclass AUC
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
        except Exception:
            metrics["roc_auc"] = None

        return metrics

    # OOF predictions on TRAIN
    y_oof = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict", n_jobs=n_jobs
    )
    try:
        y_oof_proba = cross_val_predict(
            pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=n_jobs
        )
    except Exception:
        y_oof_proba = None

    # OOF metrics and report
    oof_metrics = compute_metrics(y_train, y_oof, y_oof_proba)
    print("\n=== CV OOF Performance (TRAIN) ===")
    print(classification_report(y_train, y_oof, zero_division=0))
    print(
        f"Accuracy: {oof_metrics['accuracy']:.4f}, "
        f"Precision_w: {oof_metrics['precision_weighted']:.4f}, "
        f"Recall_w: {oof_metrics['recall_weighted']:.4f}, "
        f"F1_w: {oof_metrics['f1_weighted']:.4f}, "
        f"ROC AUC: {oof_metrics['roc_auc'] if oof_metrics['roc_auc'] is not None else 'N/A'}"
    )

    # Fit on TRAIN and evaluate on held-out TEST
    fitted_pipe = pipe.fit(X_train, y_train)
    y_test_pred = fitted_pipe.predict(X_test)
    try:
        y_test_proba = fitted_pipe.predict_proba(X_test)
    except Exception:
        y_test_proba = None

    # TEST metrics and report
    test_metrics = compute_metrics(
        y_test,
        y_test_pred,
        y_test_proba,
        class_order=getattr(fitted_pipe, "classes_", None),
    )
    print("\n=== Held-out TEST Performance ===")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    print(
        f"Accuracy: {test_metrics['accuracy']:.4f}, "
        f"Precision_w: {test_metrics['precision_weighted']:.4f}, "
        f"Recall_w: {test_metrics['recall_weighted']:.4f}, "
        f"F1_w: {test_metrics['f1_weighted']:.4f}, "
        f"ROC AUC: {test_metrics['roc_auc'] if test_metrics['roc_auc'] is not None else 'N/A'}"
    )

    results = {"oof": oof_metrics, "test": test_metrics}
    return fitted_pipe, results, y_oof, y_oof_proba, test_metrics


def evaluate_model(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    y_train: pd.Series,
    y_pred_train: np.ndarray,
    y_proba_train: np.ndarray,
):
    """Evaluate the model performance with plots and summary table."""

    def get_metrics(y_true, y_pred, y_proba):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        return accuracy, precision, recall, f1, roc_auc

    # Test set metrics
    accuracy, precision, recall, f1, roc_auc = get_metrics(y_test, y_pred, y_proba)

    # Train set metrics
    (
        train_accuracy,
        train_precision,
        train_recall,
        train_f1,
        train_roc_auc,
    ) = get_metrics(y_train, y_pred_train, y_proba_train)

    print("=== Train Set Classification Report ===")
    print(classification_report(y_train, y_pred_train, zero_division=0))

    print("=== Test Set Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("=== Train Metrics ===")
    print(
        f"Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}, ROC AUC: {train_roc_auc:.4f}"
    )

    print("=== Test Metrics ===")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, ROC AUC: {roc_auc:.4f}"
    )

    print("=== Confusion Matrix ===")
    # Plot confusion matrices for train and test
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train confusion matrix
    cm_train = confusion_matrix(y_train, y_pred_train)
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Train Confusion Matrix")

    # Test confusion matrix
    cm_test = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Test Confusion Matrix")

    plt.tight_layout()
    plt.show()

    print("=== ROC AUC Curve ===")
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train[:, 1])
    fpr_test, tpr_test, _ = roc_curve(y_test, y_proba[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC curve (AUC = {train_roc_auc:.4f})")
    plt.plot(fpr_test, tpr_test, label=f"Test ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Train & Test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
