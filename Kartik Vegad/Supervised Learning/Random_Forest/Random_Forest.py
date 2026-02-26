# =============================================================================
# CREDIT DEFAULT PREDICTION - RANDOM FOREST (FULL ARCHITECTED VERSION)
# =============================================================================

import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = r"credit_default.csv"
MODEL_PATH = r"models\random_forest_model.pkl"
GRAPH_DIR = r"graphs"

RANDOM_STATE = 42
TEST_SIZE = 0.2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

os.makedirs(GRAPH_DIR, exist_ok=True)


# =============================================================================
# METRICS STRUCTURE
# =============================================================================

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


# =============================================================================
# DATASET LOADER
# =============================================================================

class DatasetLoader:

    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        logger.info("Loading dataset...")

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset not found at {self.path}")

        df = pd.read_csv(self.path, skiprows=1)
        logger.info(f"Dataset shape: {df.shape}")

        return df


# =============================================================================
# DATASET VALIDATOR
# =============================================================================

class DatasetValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate(self) -> bool:
        logger.info("Running dataset validation...")

        if self.df.empty:
            logger.error("Dataset is empty.")
            return False

        if "default payment next month" not in self.df.columns:
            logger.error("Target column missing.")
            return False

        missing = self.df.isnull().sum().sum()
        if missing > 0:
            logger.warning(f"Dataset contains {missing} missing values.")

        logger.info("Dataset validation completed.")
        return True


# =============================================================================
# DATASET PROCESSOR
# =============================================================================

class DatasetProcessor:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def process(self) -> Tuple[pd.DataFrame, pd.Series]:

        if "ID" in self.df.columns:
            self.df.drop(columns=["ID"], inplace=True)

        X = self.df.drop(columns=["default payment next month"])
        y = self.df["default payment next month"]

        return X, y


# =============================================================================
# DATASET VISUALIZER
# =============================================================================

class DatasetVisualizer:

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def visualize(self):

        self._plot_target_distribution()
        self._plot_correlation_heatmap()

    def _plot_target_distribution(self):
        plt.figure()
        self.y.value_counts().plot(kind="bar")
        plt.title("Target Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, "target_distribution.png"))
        plt.close()

    def _plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.X.corr(), cmap="coolwarm")
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, "correlation_heatmap.png"))
        plt.close()


# =============================================================================
# RANDOM FOREST MODEL
# =============================================================================

class RandomForestModel:

    def __init__(self):
        self.search: Optional[RandomizedSearchCV] = None

    def build(self, X: pd.DataFrame) -> RandomizedSearchCV:

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features)
            ],
            remainder="passthrough"
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                ))
            ]
        )

        param_dist = {
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__class_weight": [None, "balanced"]
        }

        self.search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="f1",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        return self.search


# =============================================================================
# MODEL EVALUATOR
# =============================================================================

class ModelEvaluator:

    def __init__(self, model: Pipeline):
        self.model = model

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        metrics = ModelMetrics(
            accuracy=accuracy_score(y, preds),
            precision=precision_score(y, preds, zero_division=0),
            recall=recall_score(y, preds, zero_division=0),
            f1=f1_score(y, preds, zero_division=0),
            roc_auc=roc_auc_score(y, probs)
        )

        logger.info(metrics)
        self._plot_confusion_matrix(X, y)

        return metrics

    def _plot_confusion_matrix(self, X, y):
        cm = confusion_matrix(y, self.model.predict(X))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
        plt.close()


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class MLPipeline:

    def __init__(self):
        self.loader: Optional[DatasetLoader] = None
        self.validator: Optional[DatasetValidator] = None
        self.processor: Optional[DatasetProcessor] = None
        self.visualizer: Optional[DatasetVisualizer] = None
        self.model_builder: Optional[RandomForestModel] = None
        self.evaluator: Optional[ModelEvaluator] = None

    def run(self):

        # Load
        self.loader = DatasetLoader(DATASET_PATH)
        df = self.loader.load()

        # Validate
        self.validator = DatasetValidator(df)
        if not self.validator.validate():
            raise ValueError("Dataset validation failed.")

        # Process
        self.processor = DatasetProcessor(df)
        X, y = self.processor.process()

        # Visualize
        self.visualizer = DatasetVisualizer(X, y)
        self.visualizer.visualize()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )

        # Train
        self.model_builder = RandomForestModel()
        search = self.model_builder.build(X)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Evaluate
        self.evaluator = ModelEvaluator(best_model)
        metrics = self.evaluator.evaluate(X_test, y_test)

        # Save
        artifact: Dict[str, Any] = {
            "model": best_model,
            "metrics": metrics,
            "best_params": search.best_params_
        }

        joblib.dump(artifact, MODEL_PATH)
        logger.info("Model saved successfully.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pipeline = MLPipeline()
    pipeline.run()