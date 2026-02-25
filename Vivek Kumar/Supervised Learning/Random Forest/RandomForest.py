import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import seaborn as sns
import joblib
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ─── Global Variables ─────────────────────────────────────────────────────────
RANDOM_STATE    = 42
DATASET_PATH    = None          # Set to a CSV path to load from disk; None → sklearn built-in
TEST_SIZE       = 0.2

# Random Forest Hyperparameters
N_ESTIMATORS       = 100        # Number of trees in the forest
MAX_DEPTH          = None       # None → grow until leaves are pure
MIN_SAMPLES_SPLIT  = 2          # Min samples required to split an internal node
MIN_SAMPLES_LEAF   = 1          # Min samples required to be at a leaf node
MAX_FEATURES       = 'sqrt'     # 'sqrt' | 'log2' | int | float
BOOTSTRAP          = True       # Use bootstrap samples when building trees
CLASS_WEIGHT       = None       # None | 'balanced' | 'balanced_subsample'

# Model Persistence
MODEL_SAVE_PATH = "random_forest_iris.pkl"

# Visualization Configuration
DPI   = 100
STYLE = 'seaborn-v0_8-darkgrid'


# ─── Data Class ───────────────────────────────────────────────────────────────
@dataclass
class ModelMetrics:
    """Stores model evaluation metrics."""
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float

    def __str__(self) -> str:
        return (
            f"Model Performance Metrics:\n"
            f"{'=' * 50}\n"
            f"Accuracy:   {self.accuracy:.4f}\n"
            f"Precision:  {self.precision:.4f}\n"
            f"Recall:     {self.recall:.4f}\n"
            f"F1-Score:   {self.f1:.4f}\n"
            f"ROC-AUC:    {self.roc_auc:.4f}\n"
            f"{'=' * 50}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════════
class DatasetLoader:
    """Loads the Iris dataset from CSV on disk or from sklearn's built-in."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path   = dataset_path
        self.data:          Optional[pd.DataFrame] = None
        self.target:        Optional[pd.Series]    = None
        self.feature_names: Optional[list]          = None
        self.class_names:   Optional[list]          = None

    def _load_builtin(self) -> Tuple[pd.DataFrame, pd.Series]:
        iris = load_iris()
        self.feature_names = list(iris.feature_names)
        self.class_names   = list(iris.target_names)
        self.data   = pd.DataFrame(iris.data, columns=self.feature_names)
        self.target = pd.Series(iris.target, name='species')
        return self.data, self.target

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                if 'species' in df.columns:
                    self.data   = df.drop(columns=['species'])
                    self.target = df['species']
                else:
                    self.data   = df.iloc[:, :-1]
                    self.target = df.iloc[:, -1]
                self.feature_names = list(self.data.columns)
                self.class_names   = [str(c) for c in sorted(self.target.unique())]
                print(f"✓ Loaded from: {self.dataset_path}")
            except Exception as ex:
                print(f"⚠ Failed to load {self.dataset_path}: {ex}")
                print("⚠ Falling back to sklearn built-in Iris dataset")
                self._load_builtin()
        else:
            self._load_builtin()
            print("✓ Loaded built-in sklearn Iris dataset")

        print(f"✓ Samples      : {len(self.data)}")
        print(f"✓ Features     : {len(self.feature_names)}")
        print(f"✓ Feature names: {', '.join(self.feature_names)}")
        print(f"✓ Classes      : {self.class_names}")
        return self.data, self.target


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATASET VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════
class DatasetValidator:
    """Validates dataset integrity before processing."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        passed = True

        if self.data.empty or self.target.empty:
            print("✗ ERROR: Dataset is empty!")
            return False
        print("✓ Dataset is not empty")

        print(f"\n--- Dataset Shape ---")
        print(f"Features shape : {self.data.shape}")
        print(f"Target shape   : {self.target.shape}")

        if self.data.shape[0] != self.target.shape[0]:
            print("✗ ERROR: Features and target row count mismatch!")
            return False
        print("✓ Row counts match")

        print(f"\n--- Missing Values ---")
        missing_X = self.data.isnull().sum().sum()
        missing_y = self.target.isnull().sum()
        print(f"Features : {missing_X}   Target : {missing_y}")
        if missing_X > 0 or missing_y > 0:
            print("⚠ WARNING: Missing values detected — will be imputed in processing")
            passed = False
        else:
            print("✓ No missing values")

        print(f"\n--- Data Types ---")
        print(self.data.dtypes)
        non_numeric = self.data.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"⚠ WARNING: Non-numeric columns: {list(non_numeric)}")
            passed = False
        else:
            print("✓ All features are numeric")

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Statistical Summary ---")
        print(self.data.describe())

        print(f"\n--- Class Distribution ---")
        counts = self.target.value_counts()
        print(counts)
        print(f"Balance ratio: {counts.min() / counts.max():.2f}")

        inf_count = np.isinf(self.data.select_dtypes(include=[np.number]).values).sum()
        if inf_count > 0:
            print(f"⚠ WARNING: {inf_count} infinite values")
            passed = False
        else:
            print("✓ No infinite values")

        return passed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════
class DatasetProcessor:
    """Handles imputation and feature standardization.

    NOTE: Random Forest is inherently scale-invariant (tree splits on rank,
    not magnitude), so standardization is NOT strictly required. We apply
    it here for pipeline consistency; it does not harm RF performance.
    """

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data.copy()
        self.target = target.copy()
        self.feature_means: Optional[pd.Series] = None
        self.feature_stds:  Optional[pd.Series] = None
        self.processed_data:   Optional[pd.DataFrame] = None
        self.processed_target: Optional[pd.Series]    = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        print("\n--- Handling Missing Values ---")
        missing_before = self.data.isnull().sum().sum()
        self.data   = self.data.fillna(self.data.mean(numeric_only=True))
        self.target = self.target.fillna(self.target.mode()[0])
        print(f"Missing before: {missing_before}  →  after: {self.data.isnull().sum().sum()}")

        print("\n--- Feature Standardization (Z-Score) ---")
        print("  ⓘ  RF is scale-invariant; standardization applied for pipeline consistency.")
        self.feature_means = self.data.mean()
        self.feature_stds  = self.data.std().replace(0, 1.0)
        self.processed_data = (self.data - self.feature_means) / self.feature_stds
        print("✓ Features standardized (mean≈0, std≈1)")

        print("\n--- Target Encoding ---")
        self.processed_target = self.target.astype(int)
        print(f"Target classes: {sorted(self.processed_target.unique())}")

        print(f"\n--- Processed Shape ---")
        print(f"Features : {self.processed_data.shape}   Target: {self.processed_target.shape}")

        return self.processed_data, self.processed_target


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IRIS VISUALIZER
# ═══════════════════════════════════════════════════════════════════════════════
class IrisVisualizer:
    """10-plot visualization suite for the Iris dataset."""

    COLORS = ['steelblue', 'darkorange', 'forestgreen']
    LABELS = ['Setosa (0)', 'Versicolor (1)', 'Virginica (2)']

    def __init__(self, data: pd.DataFrame, target: pd.Series,
                 class_names: Optional[list] = None):
        self.data        = data
        self.target      = target
        self.class_names = class_names or ['Setosa', 'Versicolor', 'Virginica']
        plt.style.use(STYLE)

    # ──────────────────────────────────────────────────────────────────
    def visualize(self):
        print(f"\n{'=' * 70}")
        print("IRIS DATASET VISUALIZATION")
        print(f"{'=' * 70}")

        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_feature_distributions()
        self.plot_feature_boxplots()
        self.plot_pairplot()
        self.plot_feature_violin_plots()
        self.plot_feature_kde_plots()
        self.plot_feature_statistics()
        self.plot_feature_importance_proxy()
        self.plot_3d_scatter()

        print("✓ All Iris visualizations saved")

    # 1️⃣  CLASS DISTRIBUTION ─────────────────────────────────────────
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_index()
        labels = [self.class_names[i] for i in counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        bars = axes[0].bar(labels, counts.values,
                           color=self.COLORS, edgecolor='black', alpha=0.7)
        axes[0].set_title("Iris Species Distribution", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., h,
                         f'{int(h)}', ha='center', va='bottom')

        axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
                    colors=self.COLORS, startangle=90,
                    explode=[0.05] * len(counts))
        axes[1].set_title("Class Ratio", fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("iris_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Target distribution saved")

    # 2️⃣  CORRELATION HEATMAP ─────────────────────────────────────────
    def plot_correlation_heatmap(self):
        df = self.data.copy()
        df['species'] = self.target
        plt.figure(figsize=(10, 8), dpi=DPI)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f',
                    center=0, square=True, linewidths=1,
                    cbar_kws={'shrink': 0.8})
        plt.title("Iris Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("iris_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")

    # 3️⃣  FEATURE DISTRIBUTIONS ───────────────────────────────────────
    def plot_feature_distributions(self):
        n = len(self.data.columns)
        cols, rows = 2, (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            for cls, color, label in zip(sorted(self.target.unique()),
                                         self.COLORS, self.LABELS):
                axes[i].hist(self.data[self.target == cls][col], bins=20,
                             alpha=0.6, label=label, color=color, edgecolor='black')
            axes[i].set_title(col, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("iris_feature_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature distributions saved")

    # 4️⃣  BOXPLOTS ────────────────────────────────────────────────────
    def plot_feature_boxplots(self):
        # seaborn sometimes casts categorical levels to strings, so ensure our
        # palette keys are strings to avoid missing-key errors.
        palette = dict(zip(map(str, sorted(self.target.unique())), self.COLORS))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            sns.boxplot(x=self.target, y=self.data[col],
                        ax=axes[i], palette=palette)
            axes[i].set_title(f"{col} by Species", fontsize=11, fontweight='bold')
            axes[i].set_xticklabels(self.class_names)
            axes[i].set_xlabel('Species')
            axes[i].set_ylabel(col)
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("iris_feature_boxplots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature boxplots saved")

    # 5️⃣  PAIRPLOT (Iris-specific) ─────────────────────────────────────
    def plot_pairplot(self):
        df = self.data.copy()
        df['Species'] = self.target.map(
            {i: name for i, name in enumerate(self.class_names)}
        )
        g = sns.pairplot(df, hue='Species',
                         palette=dict(zip(self.class_names, self.COLORS)),
                         diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle("Iris Feature Pairplot", y=1.02,
                        fontsize=14, fontweight='bold')
        plt.savefig("iris_pairplot.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Pairplot saved")

    # 6️⃣  VIOLIN PLOTS ────────────────────────────────────────────────
    def plot_feature_violin_plots(self):
        n = len(self.data.columns)
        cols, rows = 2, (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4), dpi=DPI)
        axes = axes.flatten()
        # ensure palette keys are strings to align with how seaborn handles
        # the 'species' column values
        palette = dict(zip(map(str, sorted(self.target.unique())), self.COLORS))

        plot_df = self.data.copy()
        plot_df['species'] = self.target

        for i, col in enumerate(self.data.columns):
            sns.violinplot(data=plot_df, x='species', y=col,
                           ax=axes[i], palette=palette, alpha=0.7)
            axes[i].set_title(f"{col} by Species", fontsize=11, fontweight='bold')
            axes[i].set_xticklabels(self.class_names)
            axes[i].set_xlabel('Species')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3, axis='y')

        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("iris_feature_violin_plots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Violin plots saved")

    # 7️⃣  KDE PLOTS ───────────────────────────────────────────────────
    def plot_feature_kde_plots(self):
        n = len(self.data.columns)
        cols, rows = 2, (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(self.data.columns):
            for cls, color, label in zip(sorted(self.target.unique()),
                                         self.COLORS, self.LABELS):
                self.data[self.target == cls][col].plot.kde(
                    ax=axes[i], linewidth=2,
                    label=label, color=color, alpha=0.8)
            axes[i].set_title(f"{col} — KDE", fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Density')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)

        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("iris_feature_kde_plots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] KDE plots saved")

    # 8️⃣  FEATURE STATISTICS HEATMAP ──────────────────────────────────
    def plot_feature_statistics(self):
        stats = {}
        for col in self.data.columns:
            for cls, name in zip(sorted(self.target.unique()), self.class_names):
                subset = self.data[self.target == cls][col]
                stats.setdefault(f"{col}\n(Mean)", {})[name] = subset.mean()
                stats.setdefault(f"{col}\n(Std)",  {})[name] = subset.std()

        stats_df = pd.DataFrame(stats).T
        norm_df  = (stats_df - stats_df.values.min()) / \
                   (stats_df.values.max() - stats_df.values.min())

        fig, axes = plt.subplots(1, 2, figsize=(18, 5), dpi=DPI)
        sns.heatmap(stats_df, annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=axes[0], cbar_kws={'shrink': 0.8})
        axes[0].set_title('Feature Statistics by Class (Raw)',
                          fontsize=12, fontweight='bold')

        sns.heatmap(norm_df, annot=stats_df.values, fmt='.2f', cmap='RdYlGn',
                    ax=axes[1], cbar_kws={'shrink': 0.8})
        axes[1].set_title('Feature Statistics by Class (Normalized)',
                          fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("iris_feature_statistics.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature statistics heatmap saved")

    # 9️⃣  FEATURE IMPORTANCE PROXY (correlation-based) ─────────────────
    def plot_feature_importance_proxy(self):
        """Correlation with target as a quick importance proxy (pre-training)."""
        # One-hot encode target for multi-class correlation proxy
        target_dummies = pd.get_dummies(self.target)
        corr_vals = pd.Series({
            col: target_dummies.corrwith(self.data[col]).abs().mean()
            for col in self.data.columns
        }).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI)
        colors = ['#d62728' if v == corr_vals.max() else '#1f77b4'
                  for v in corr_vals.values]
        bars = ax.barh(corr_vals.index, corr_vals.values,
                       color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Mean Absolute Correlation with Classes')
        ax.set_title('Feature Importance Proxy (Pre-Training Correlation)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, corr_vals.values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2.,
                    f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig("iris_feature_importance_proxy.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature importance proxy saved")

    # 🔟  3D SCATTER ────────────────────────────────────────────────────
    def plot_3d_scatter(self):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        top3 = self.data.var().sort_values(ascending=False).head(3).index.tolist()
        if len(top3) < 3:
            print("[SKIP] 3D scatter: fewer than 3 features")
            return

        fig = plt.figure(figsize=(12, 9), dpi=DPI)
        ax  = fig.add_subplot(111, projection='3d')

        for cls, color, label in zip(sorted(self.target.unique()),
                                     self.COLORS, self.LABELS):
            mask = self.target == cls
            ax.scatter(self.data[mask][top3[0]],
                       self.data[mask][top3[1]],
                       self.data[mask][top3[2]],
                       c=color, label=label, s=40, alpha=0.7,
                       edgecolors='k', linewidths=0.3)

        ax.set_xlabel(top3[0], fontsize=10, fontweight='bold')
        ax.set_ylabel(top3[1], fontsize=10, fontweight='bold')
        ax.set_zlabel(top3[2], fontsize=10, fontweight='bold')
        ax.set_title('3D Visualization — Top 3 Variance Features',
                     fontsize=13, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig("iris_3d_scatter.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 3D scatter plot saved")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RANDOM FOREST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class RandomForestModel:
    """
    Random Forest classifier wrapper for Iris species prediction.

    Encapsulates training, prediction, probability estimation,
    feature importance, and single-model evaluation.
    """

    def __init__(
        self,
        n_estimators:      int           = N_ESTIMATORS,
        max_depth:         Optional[int] = MAX_DEPTH,
        min_samples_split: int           = MIN_SAMPLES_SPLIT,
        min_samples_leaf:  int           = MIN_SAMPLES_LEAF,
        max_features:      str           = MAX_FEATURES,
        bootstrap:         bool          = BOOTSTRAP,
        class_weight                     = CLASS_WEIGHT,
    ):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.bootstrap         = bootstrap
        self.class_weight      = class_weight

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    # ── Training ──────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        print(f"\n{'=' * 60}")
        print("TRAINING RANDOM FOREST (IRIS)")
        print(f"{'=' * 60}")
        print(f"n_estimators  : {self.n_estimators}")
        print(f"max_depth     : {self.max_depth}")
        print(f"max_features  : {self.max_features}")
        print(f"bootstrap     : {self.bootstrap}")

        self.model.fit(X_train, y_train)
        print("✓ Training completed")

        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            print(f"✓ Model saved → {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"⚠ Could not save model: {ex}")

    # ── Inference ─────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return full probability matrix (n_samples × n_classes)."""
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    # ── Quick self-evaluation ──────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y_true: np.ndarray,
                 name: str = "Test Set") -> dict:
        print(f"\n{'=' * 60}")
        print(f"RANDOM FOREST EVALUATION — {name}")
        print(f"{'=' * 60}")

        y_pred  = self.predict(X)
        y_proba = self.predict_proba(X)
        avg     = 'macro'

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
        recall    = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1        = f1_score(y_true, y_pred, average=avg, zero_division=0)
        roc_auc   = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}  (macro avg)")
        print(f"Recall    : {recall:.4f}  (macro avg)")
        print(f"F1 Score  : {f1:.4f}  (macro avg)")
        print(f"ROC-AUC   : {roc_auc:.4f}  (OvR macro avg)")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Setosa', 'Versicolor', 'Virginica']))

        return dict(accuracy=accuracy, precision=precision,
                    recall=recall, f1=f1, roc_auc=roc_auc)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MODEL EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════
class ModelEvaluator:
    """Generates comprehensive evaluation plots for the Random Forest model."""

    CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']
    COLORS      = ['steelblue', 'darkorange', 'forestgreen']

    def __init__(self, model: RandomForestModel):
        self.model = model

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y_true: np.ndarray,
                 dataset_name: str = "Dataset",
                 feature_names: Optional[list] = None) -> None:
        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATOR — {dataset_name}")
        print(f"{'=' * 70}")

        y_pred  = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        avg     = 'macro'

        print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision : {precision_score(y_true, y_pred, average=avg, zero_division=0):.4f}")
        print(f"Recall    : {recall_score(y_true, y_pred, average=avg, zero_division=0):.4f}")
        print(f"F1 Score  : {f1_score(y_true, y_pred, average=avg, zero_division=0):.4f}")
        print(f"ROC AUC   : {roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'):.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.CLASS_NAMES))

        self._plot_evaluation(y_true, y_pred, y_proba, dataset_name)
        self._plot_prediction_distribution(y_true, y_pred, y_proba, dataset_name)
        if feature_names:
            self._plot_feature_importance(feature_names, dataset_name)

    # ------------------------------------------------------------------
    def _plot_evaluation(self, y_true, y_pred, y_proba, dataset_name):
        from sklearn.preprocessing import label_binarize

        n_classes = y_proba.shape[1]
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=DPI)

        # Confusion Matrix
        cm      = confusion_matrix(y_true, y_pred)
        cm_pct  = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot   = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
                             for j in range(cm.shape[1])]
                            for i in range(cm.shape[0])])
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[0, 0],
                    xticklabels=self.CLASS_NAMES,
                    yticklabels=self.CLASS_NAMES,
                    cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title(f'Confusion Matrix — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # ROC Curves (One-vs-Rest)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for cls, color, name in zip(range(n_classes), self.COLORS, self.CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_bin[:, cls], y_proba[:, cls])
            auc = roc_auc_score(y_bin[:, cls], y_proba[:, cls])
            axes[0, 1].plot(fpr, tpr, linewidth=2, color=color,
                            label=f'{name} (AUC={auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curves (OvR) — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='lower right', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        # Predicted Probability Distributions
        for cls, color, name in zip(range(n_classes), self.COLORS, self.CLASS_NAMES):
            axes[1, 0].hist(y_proba[y_true == cls, cls], bins=15,
                            alpha=0.6, color=color, label=name, edgecolor='black')
        axes[1, 0].set_xlabel('Predicted Probability (correct class)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Probability Distribution — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Metrics Bar Chart
        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_vals  = [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average='macro', zero_division=0),
            recall_score(y_true, y_pred, average='macro', zero_division=0),
            f1_score(y_true, y_pred, average='macro', zero_division=0),
            roc_auc,
        ]
        bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[1, 1].bar(metric_names, metric_vals,
                              color=bar_colors, edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title(f'Performance Metrics — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, metric_vals):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        fname = f"evaluation_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plots saved → {fname}")

    # ------------------------------------------------------------------
    def _plot_prediction_distribution(self, y_true, y_pred, y_proba, dataset_name):
        """Confidence analysis + calibration reliability diagram."""
        from sklearn.preprocessing import label_binarize

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
        n_classes = y_proba.shape[1]

        # Correct vs Incorrect confidence
        max_proba  = y_proba.max(axis=1)
        is_correct = y_true == y_pred
        axes[0].hist(max_proba[is_correct],  bins=20, alpha=0.6,
                     color='forestgreen', label='Correct', edgecolor='black')
        axes[0].hist(max_proba[~is_correct], bins=20, alpha=0.6,
                     color='crimson',      label='Incorrect', edgecolor='black')
        axes[0].set_xlabel('Max Predicted Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Prediction Confidence — {dataset_name}',
                          fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Calibration plot (macro over all classes)
        y_bin  = label_binarize(y_true, classes=list(range(n_classes)))
        edges  = np.linspace(0, 1, 11)
        all_pred, all_true = [], []
        for cls in range(n_classes):
            mp, mt = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (y_proba[:, cls] >= lo) & (y_proba[:, cls] < hi)
                if mask.sum() > 0:
                    mp.append(y_proba[mask, cls].mean())
                    mt.append(y_bin[mask, cls].mean())
            if mp:
                all_pred.append(mp)
                all_true.append(mt)

        if all_pred:
            min_len  = min(len(p) for p in all_pred)
            mean_pred = np.mean([p[:min_len] for p in all_pred], axis=0)
            mean_true = np.mean([t[:min_len] for t in all_true], axis=0)
            axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
            axes[1].plot(mean_pred, mean_true, 'o-', linewidth=2,
                         markersize=8, color='steelblue', label='Model Calibration')
            axes[1].fill_between(mean_pred, mean_true, mean_pred, alpha=0.15)

        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Actual Positive Rate')
        axes[1].set_title(f'Calibration Plot — {dataset_name}',
                          fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"predictions_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Prediction distribution plots saved → {fname}")

    # ------------------------------------------------------------------
    def _plot_feature_importance(self, feature_names: list, dataset_name: str):
        """Gini-based feature importance from the trained forest."""
        importances = self.model.get_feature_importances()
        indices     = np.argsort(importances)   # ascending for horizontal bar
        sorted_names = [feature_names[i] for i in indices]
        sorted_imp   = importances[indices]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
        colors = ['#d62728' if v == sorted_imp.max() else '#1f77b4'
                  for v in sorted_imp]
        bars = ax.barh(sorted_names, sorted_imp,
                       color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Gini Feature Importance')
        ax.set_title(f'Random Forest Feature Importance — {dataset_name}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, sorted_imp):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2.,
                    f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        fname = f"feature_importance_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Feature importance saved → {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ML PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
class MLPipeline:
    """End-to-end pipeline: Load → Validate → Visualize → Process → Train → Evaluate."""

    def __init__(self):
        self.loader:    DatasetLoader              = DatasetLoader(DATASET_PATH)
        self.processor: Optional[DatasetProcessor] = None
        self.model:     Optional[RandomForestModel] = None
        self.evaluator: Optional[ModelEvaluator]   = None

    # ------------------------------------------------------------------
    def run(self):
        print("\n" + "=" * 70)
        print("RANDOM FOREST PIPELINE — IRIS SPECIES CLASSIFICATION")
        print("=" * 70)

        # 1️⃣  Load
        data, target = self.loader.load_data()

        # 2️⃣  Validate
        validator = DatasetValidator(data, target)
        if not validator.verify_dataset():
            raise RuntimeError("Dataset validation failed — aborting.")

        # 3️⃣  Visualize raw data
        visualizer = IrisVisualizer(data, target,
                                    class_names=self.loader.class_names)
        visualizer.visualize()

        # 4️⃣  Process
        self.processor = DatasetProcessor(data, target)
        processed_data, processed_target = self.processor.process_dataset()

        # 5️⃣  Split
        X_train, X_test, y_train, y_test = train_test_split(
            processed_data.values,
            processed_target.values,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=processed_target,
        )

        # 6️⃣  Train
        self.model = RandomForestModel()
        self.model.fit(X_train, y_train)

        # 7️⃣  Evaluate
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set",
                                feature_names=self.loader.feature_names)
        self.evaluator.evaluate(X_test, y_test, "Test Set",
                                feature_names=self.loader.feature_names)

        # 8️⃣  Cross-Validation
        self._perform_cross_validation(processed_data.values,
                                       processed_target.values)

        # 9️⃣  Predict new sample
        self._predict_new_sample(X_train.shape[1])

    # ------------------------------------------------------------------
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stratified 5-fold cross-validation."""
        print(f"\n{'=' * 70}")
        print("K-FOLD CROSS-VALIDATION (5-Fold, Stratified)")
        print(f"{'=' * 70}")

        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        f1_scores  = cross_val_score(clf, X, y, cv=cv, scoring='f1_macro')
        roc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc_ovr')

        print(f"\nAccuracy  : {acc_scores.mean():.4f}  (+/- {acc_scores.std():.4f})")
        print(f"F1-Macro  : {f1_scores.mean():.4f}  (+/- {f1_scores.std():.4f})")
        print(f"ROC-AUC   : {roc_scores.mean():.4f}  (+/- {roc_scores.std():.4f})")

        stability = ('GOOD generalization ✓'
                     if acc_scores.std() < 0.05
                     else 'HIGH variance ⚠ — consider tuning')
        print(f"\n✓ Cross-validation indicates {stability}")

    # ------------------------------------------------------------------
    def _predict_new_sample(self, n_features: int) -> None:
        """Demonstrate inference on a synthetic new observation."""
        print(f"\n{'=' * 70}")
        print("NEW SAMPLE PREDICTION")
        print(f"{'=' * 70}")

        means = self.processor.feature_means.values
        stds  = self.processor.feature_stds.values
        raw   = means + np.random.randn(n_features) * stds
        scaled = ((raw - means) / stds).reshape(1, -1)

        proba = self.model.predict_proba(scaled)[0]
        pred  = self.model.predict(scaled)[0]
        names = self.loader.class_names or ['Setosa', 'Versicolor', 'Virginica']

        print(f"Predicted class : {names[pred]} (class {pred})")
        for name, p in zip(names, proba):
            print(f"  P({name:>12s}) = {p:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Run the Random Forest Iris classification pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as exc:
        print(f"\n❌ Pipeline error: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
