import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import joblib
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
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ─── Global Variables ──────────────────────────────────────────────────────────
RANDOM_STATE  = 42
DATASET_PATH  = "play_tennis.csv"
TEST_SIZE     = 0.2
VALIDATION_SIZE = 0.1

# Naive Bayes Hyperparameter
ALPHA = 1.0   # Laplace smoothing (0 = no smoothing, 1 = Laplace)

# Model Persistence
MODEL_SAVE_PATH = "nb_tennis_model.pkl"

# Visualization
DPI   = 100
STYLE = 'seaborn-v0_8-darkgrid'

# ─── Play Tennis Feature Space ─────────────────────────────────────────────────
OUTLOOK_VALUES  = ['Sunny', 'Overcast', 'Rain']
TEMP_VALUES     = ['Hot', 'Mild', 'Cool']
HUMIDITY_VALUES = ['High', 'Normal']
WIND_VALUES     = ['Weak', 'Strong']


# ─── Data Class ───────────────────────────────────────────────────────────────
@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════
class DatasetLoader:
    """
    Loads the Play Tennis CSV or generates 800-row synthetic data.

    Synthetic generation rule (mirrors classic decision logic):
      Overcast            → ~85 % Yes
      Sunny + High        → ~15 % Yes
      Sunny + Normal      → ~85 % Yes
      Rain  + Strong Wind → ~15 % Yes
      Rain  + Weak Wind   → ~85 % Yes
      Hot temperature     → -5 % Yes adjustment
      Cool temperature    → +5 % Yes adjustment
    """

    def __init__(self, dataset_path: str = None):
        self.dataset_path  = dataset_path
        self.data:         Optional[pd.DataFrame] = None
        self.target:       Optional[pd.Series]    = None
        self.feature_names: Optional[list]        = None

    # ------------------------------------------------------------------
    def _generate_synthetic_dataset(self, n_samples: int = 800) -> Tuple[pd.DataFrame, pd.Series]:
        np.random.seed(RANDOM_STATE)

        outlooks   = np.random.choice(OUTLOOK_VALUES,  n_samples, p=[0.35, 0.30, 0.35])
        temps      = np.random.choice(TEMP_VALUES,     n_samples, p=[0.30, 0.40, 0.30])
        humidities = np.random.choice(HUMIDITY_VALUES, n_samples, p=[0.50, 0.50])
        winds      = np.random.choice(WIND_VALUES,     n_samples, p=[0.55, 0.45])

        play = []
        for o, t, h, w in zip(outlooks, temps, humidities, winds):
            if o == 'Overcast':
                p_yes = 0.85
            elif o == 'Sunny':
                p_yes = 0.15 if h == 'High' else 0.85
            else:  # Rain
                p_yes = 0.15 if w == 'Strong' else 0.85

            if t == 'Hot':
                p_yes -= 0.05
            elif t == 'Cool':
                p_yes += 0.05
            p_yes = np.clip(p_yes, 0.05, 0.95)

            play.append('Yes' if np.random.rand() < p_yes else 'No')

        df = pd.DataFrame({
            'outlook':  outlooks,
            'temp':     temps,
            'humidity': humidities,
            'wind':     winds,
        })
        target = pd.Series(play, name='play')
        return df, target

    # ------------------------------------------------------------------
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET — PLAY TENNIS")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)

                # Drop 'day' column if present
                if 'day' in df.columns:
                    df.drop(columns=['day'], inplace=True)

                if 'play' in df.columns:
                    self.data   = df.drop(columns=['play'])
                    self.target = df['play']
                else:
                    self.data   = df.iloc[:, :-1]
                    self.target = df.iloc[:, -1]

                self.feature_names = list(self.data.columns)
                original_size = len(self.data)
                print(f"✓ Loaded from: {self.dataset_path}  ({original_size} rows)")

                # Extend with synthetic rows to reach 800
                if original_size < 800:
                    extra_data, extra_target = self._generate_synthetic_dataset(
                        n_samples=800 - original_size
                    )
                    self.data   = pd.concat([self.data, extra_data],     ignore_index=True)
                    self.target = pd.concat([self.target, extra_target], ignore_index=True)
                    print(f"✓ Extended with synthetic rows → total {len(self.data)} rows")

                return self.data, self.target

            except Exception as ex:
                print(f"⚠ Failed to load CSV: {ex}  → falling back to synthetic data")

        # Full synthetic dataset
        self.data, self.target = self._generate_synthetic_dataset(n_samples=800)
        self.feature_names = list(self.data.columns)

        print(f"✓ Synthetic Play Tennis dataset generated")
        print(f"✓ Samples  : {len(self.data)}")
        print(f"✓ Features : {self.feature_names}")
        print(f"✓ Classes  : No / Yes")
        return self.data, self.target


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════
class DatasetValidator:
    """Validates the Play Tennis dataset before processing."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        if self.data.empty or self.target.empty:
            print("✗ ERROR: Dataset is empty!")
            return False
        print("✓ Dataset is not empty")

        print(f"\n--- Shape ---")
        print(f"Features : {self.data.shape}")
        print(f"Target   : {self.target.shape}")

        if self.data.shape[0] != len(self.target):
            print("✗ ERROR: Row mismatch!")
            return False
        print("✓ Rows match")

        print(f"\n--- Missing Values ---")
        miss_f = self.data.isnull().sum().sum()
        miss_t = self.target.isnull().sum()
        print(f"Features: {miss_f}   Target: {miss_t}")
        if miss_f == 0 and miss_t == 0:
            print("✓ No missing values")

        print(f"\n--- Data Types ---")
        print(self.data.dtypes)

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Value Counts per Feature ---")
        for col in self.data.columns:
            print(f"  {col}: {dict(self.data[col].value_counts())}")

        print(f"\n--- Class Distribution ---")
        cc = self.target.value_counts()
        print(cc)
        print(f"Balance: {cc.min() / cc.max():.2f}")

        return True


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class DatasetProcessor:
    """
    Encodes categorical features with LabelEncoder for CategoricalNB.

    NOTE — No standardization needed for Naive Bayes:
      CategoricalNB works directly with integer-encoded categories.
      Standardization would destroy the discrete category meaning.
    """

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data.copy()
        self.target = target.copy()
        self.label_encoders: dict   = {}
        self.target_encoder         = LabelEncoder()
        self.processed_data:   Optional[pd.DataFrame] = None
        self.processed_target: Optional[pd.Series]    = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        # ── Step 1: Label-encode each categorical feature ──────────────
        print("\n--- Label Encoding Categorical Features ---")
        print("  ⓘ  CategoricalNB requires integer-encoded categories.")
        print("     No Z-score standardization needed (unlike KNN).")

        encoded_df = pd.DataFrame()
        for col in self.data.columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"  {col:12s} → {mapping}")

        self.processed_data = encoded_df

        # ── Step 2: Encode target ──────────────────────────────────────
        print("\n--- Target Encoding ---")
        self.processed_target = pd.Series(
            self.target_encoder.fit_transform(self.target.astype(str)),
            name='play'
        )
        mapping_t = dict(zip(self.target_encoder.classes_,
                             self.target_encoder.transform(self.target_encoder.classes_)))
        print(f"  Target mapping: {mapping_t}")

        print(f"\n--- Processed Shape ---")
        print(f"Features : {self.processed_data.shape}")
        print(f"Target   : {self.processed_target.shape}")

        return self.processed_data, self.processed_target


# ══════════════════════════════════════════════════════════════════════════════
# 4. VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
class TennisVisualizer:
    """10 visualizations for the Play Tennis dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target
        plt.style.use(STYLE)

    # ------------------------------------------------------------------
    def visualize(self):
        print("\n" + "=" * 70)
        print("PLAY TENNIS DATASET VISUALIZATION")
        print("=" * 70)

        self.plot_target_distribution()        # 1
        self.plot_feature_count_bars()         # 2
        self.plot_feature_vs_target()          # 3
        self.plot_correlation_heatmap()        # 4
        self.plot_stacked_bars()               # 5
        self.plot_play_rate_by_feature()       # 6
        self.plot_grouped_bars()               # 7
        self.plot_feature_statistics()         # 8
        self.plot_feature_importance()         # 9
        self.plot_pairwise_heatmaps()          # 10

        print("✓ All visualizations saved")

    # ── 1️⃣  Class Distribution ────────────────────────────────────────
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_index()

        plt.figure(figsize=(12, 5), dpi=DPI)

        plt.subplot(1, 2, 1)
        bars = plt.bar(counts.index, counts.values,
                       color=['tomato', 'mediumseagreen'], edgecolor='black', alpha=0.8)
        for b in bars:
            plt.text(b.get_x() + b.get_width()/2, b.get_height(),
                     str(int(b.get_height())), ha='center', va='bottom', fontweight='bold')
        plt.title("Play Tennis — Class Distribution", fontsize=12, fontweight='bold')
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3, axis='y')

        plt.subplot(1, 2, 2)
        plt.pie(counts.values, labels=counts.index,
                autopct='%1.1f%%', colors=['tomato', 'mediumseagreen'],
                startangle=90, explode=(0.05, 0.05))
        plt.title("Class Ratio", fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("tennis_01_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 1. Target distribution saved")

    # ── 2️⃣  Feature Count Bars ────────────────────────────────────────
    def plot_feature_count_bars(self):
        features = self.data.columns.tolist()
        fig, axes = plt.subplots(1, len(features), figsize=(16, 5), dpi=DPI)

        for ax, col in zip(axes, features):
            vc = self.data[col].value_counts()
            ax.bar(vc.index, vc.values, color='steelblue', edgecolor='black', alpha=0.8)
            for i, v in enumerate(vc.values):
                ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=15)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle("Feature Value Frequencies", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_02_feature_counts.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 2. Feature count bars saved")

    # ── 3️⃣  Feature vs Target (grouped count) ────────────────────────
    def plot_feature_vs_target(self):
        features = self.data.columns.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        df_plot = self.data.copy()
        df_plot['play'] = self.target.values

        for i, col in enumerate(features):
            ct = pd.crosstab(df_plot[col], df_plot['play'])
            ct.plot(kind='bar', ax=axes[i],
                    color=['tomato', 'mediumseagreen'], edgecolor='black', alpha=0.85)
            axes[i].set_title(f"{col} vs Play", fontsize=11, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            axes[i].tick_params(axis='x', rotation=15)
            axes[i].legend(title='Play')
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.suptitle("Feature Value Distribution by Class", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_03_feature_vs_target.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 3. Feature vs target saved")

    # ── 4️⃣  Correlation Heatmap (encoded) ────────────────────────────
    def plot_correlation_heatmap(self):
        le_target = LabelEncoder()
        df_enc = self.data.apply(lambda c: LabelEncoder().fit_transform(c.astype(str)))
        df_enc['play'] = le_target.fit_transform(self.target.astype(str))

        plt.figure(figsize=(8, 6), dpi=DPI)
        sns.heatmap(df_enc.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Heatmap (Encoded)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_04_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 4. Correlation heatmap saved")

    # ── 5️⃣  Stacked Bars ──────────────────────────────────────────────
    def plot_stacked_bars(self):
        features = self.data.columns.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        df_plot = self.data.copy()
        df_plot['play'] = self.target.values

        for i, col in enumerate(features):
            ct = pd.crosstab(df_plot[col], df_plot['play'], normalize='index') * 100
            ct.plot(kind='bar', stacked=True, ax=axes[i],
                    color=['tomato', 'mediumseagreen'], edgecolor='black', alpha=0.85)
            axes[i].set_title(f"{col} — Stacked Class %", fontsize=11, fontweight='bold')
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_ylim(0, 110)
            axes[i].tick_params(axis='x', rotation=15)
            axes[i].legend(title='Play')
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.suptitle("Stacked Class Distribution per Feature", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_05_stacked_bars.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 5. Stacked bars saved")

    # ── 6️⃣  Play Rate by Feature ──────────────────────────────────────
    def plot_play_rate_by_feature(self):
        features = self.data.columns.tolist()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        df_plot = self.data.copy()
        df_plot['play_bin'] = (self.target == 'Yes').astype(int)

        for i, col in enumerate(features):
            rate = df_plot.groupby(col)['play_bin'].mean().sort_values()
            colors = ['tomato' if v < 0.5 else 'mediumseagreen' for v in rate.values]
            bars = axes[i].bar(rate.index, rate.values * 100,
                               color=colors, edgecolor='black', alpha=0.8)
            for b in bars:
                axes[i].text(b.get_x() + b.get_width()/2, b.get_height(),
                             f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
            axes[i].axhline(50, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
            axes[i].set_title(f"Play Rate by {col}", fontsize=11, fontweight='bold')
            axes[i].set_ylabel("Play Rate (%)")
            axes[i].set_ylim(0, 110)
            axes[i].tick_params(axis='x', rotation=15)
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.suptitle("Play Rate (%) per Feature Value", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_06_play_rate.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 6. Play rate by feature saved")

    # ── 7️⃣  Grouped Bars (all features, side by side) ─────────────────
    def plot_grouped_bars(self):
        df_plot = self.data.copy()
        df_plot['play'] = self.target.values
        features = self.data.columns.tolist()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(features):
            ct = pd.crosstab(df_plot[col], df_plot['play'])
            x   = np.arange(len(ct.index))
            w   = 0.35
            axes[i].bar(x - w/2, ct.get('No',  pd.Series(0, index=ct.index)).values,
                        w, label='No',  color='tomato',        edgecolor='black', alpha=0.8)
            axes[i].bar(x + w/2, ct.get('Yes', pd.Series(0, index=ct.index)).values,
                        w, label='Yes', color='mediumseagreen', edgecolor='black', alpha=0.8)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(ct.index, rotation=15)
            axes[i].set_title(f"{col} — Grouped Count", fontsize=11, fontweight='bold')
            axes[i].set_ylabel("Count")
            axes[i].legend(title='Play')
            axes[i].grid(True, alpha=0.3, axis='y')

        plt.suptitle("Grouped Bar Charts by Feature", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_07_grouped_bars.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 7. Grouped bars saved")

    # ── 8️⃣  Feature Statistics Heatmap ───────────────────────────────
    def plot_feature_statistics(self):
        df_plot = self.data.copy()
        df_plot['play'] = self.target.values

        rows_no  = df_plot[df_plot['play'] == 'No'].drop(columns='play')
        rows_yes = df_plot[df_plot['play'] == 'Yes'].drop(columns='play')

        stats = {}
        for col in self.data.columns:
            for val in self.data[col].unique():
                stats.setdefault(col, {})[val] = {
                    'No':  (rows_no[col]  == val).sum(),
                    'Yes': (rows_yes[col] == val).sum(),
                }

        # Build a flat DataFrame for the heatmap
        records = []
        for col, vals in stats.items():
            for val, counts in vals.items():
                records.append({'Feature': col, 'Value': val,
                                'No': counts['No'], 'Yes': counts['Yes']})

        stat_df = pd.DataFrame(records).set_index(['Feature', 'Value'])

        plt.figure(figsize=(8, len(records) * 0.5 + 2), dpi=DPI)
        sns.heatmap(stat_df, annot=True, fmt='d', cmap='YlOrRd',
                    linewidths=0.5, cbar_kws={"shrink": 0.6})
        plt.title("Feature Value Counts by Class", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_08_feature_statistics.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 8. Feature statistics heatmap saved")

    # ── 9️⃣  Feature Importance (Info Gain proxy) ──────────────────────
    def plot_feature_importance(self):
        le_t = LabelEncoder()
        y_enc = le_t.fit_transform(self.target.astype(str))

        importances = {}
        for col in self.data.columns:
            le_f  = LabelEncoder()
            x_enc = le_f.fit_transform(self.data[col].astype(str))
            # Point-biserial correlation as proxy importance
            from scipy.stats import pointbiserialr
            corr, _ = pointbiserialr(x_enc, y_enc)
            importances[col] = abs(corr)

        imp_series = pd.Series(importances).sort_values(ascending=True)

        plt.figure(figsize=(8, 5), dpi=DPI)
        colors = ['#d62728' if v == imp_series.max() else '#1f77b4'
                  for v in imp_series.values]
        bars = plt.barh(imp_series.index, imp_series.values,
                        color=colors, edgecolor='black', alpha=0.8)
        for b in bars:
            plt.text(b.get_width() + 0.005, b.get_y() + b.get_height()/2,
                     f'{b.get_width():.3f}', va='center', fontsize=9)
        plt.xlabel("Abs. Correlation with Target", fontsize=11)
        plt.title("Feature Importance (Correlation Proxy)", fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig("tennis_09_feature_importance.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 9. Feature importance saved")

    # ── 🔟  Pairwise Feature Heatmaps ─────────────────────────────────
    def plot_pairwise_heatmaps(self):
        features = self.data.columns.tolist()
        pairs = [(features[i], features[j])
                 for i in range(len(features)) for j in range(i+1, len(features))]

        cols  = 3
        rows  = (len(pairs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4), dpi=DPI)
        axes  = axes.flatten()

        df_plot = self.data.copy()
        df_plot['play'] = self.target.values

        for idx, (f1, f2) in enumerate(pairs):
            ct = pd.crosstab(df_plot[f1], df_plot[f2])
            sns.heatmap(ct, annot=True, fmt='d', cmap='Blues',
                        ax=axes[idx], linewidths=0.5)
            axes[idx].set_title(f"{f1} × {f2}", fontsize=10, fontweight='bold')

        for idx in range(len(pairs), len(axes)):
            axes[idx].axis('off')

        plt.suptitle("Pairwise Feature Co-occurrence Heatmaps", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig("tennis_10_pairwise_heatmaps.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] 10. Pairwise heatmaps saved")


# ══════════════════════════════════════════════════════════════════════════════
# 5. NAIVE BAYES MODEL
# ══════════════════════════════════════════════════════════════════════════════
class NaiveBayesModel:
    """
    Categorical Naive Bayes for Play Tennis prediction.

    Why CategoricalNB?
      All features (outlook, temp, humidity, wind) are discrete categories.
      CategoricalNB estimates P(feature_value | class) for each category
      and uses Bayes' theorem:
        P(play | features) ∝ P(play) × ∏ P(feature_i | play)

    Laplace Smoothing (alpha):
      Prevents zero-probability for unseen combinations.
      alpha=1.0 → classic Laplace smoothing.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = CategoricalNB(alpha=self.alpha)

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print("\n" + "=" * 60)
        print("TRAINING NAIVE BAYES MODEL — PLAY TENNIS")
        print("=" * 60)
        print(f"Algorithm : CategoricalNB")
        print(f"Alpha     : {self.alpha}  (Laplace smoothing)")

        self.model.fit(X_train, y_train)
        print("✓ Training completed")

        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            print(f"✓ Model saved → {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"⚠ Could not save model: {ex}")

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    # ------------------------------------------------------------------
    def evaluate(self, X, y, name="Test Set"):
        print("\n" + "=" * 60)
        print(f"NB MODEL EVALUATION — {name}")
        print("=" * 60)

        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        auc  = roc_auc_score(y, y_prob)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y, y_pred))
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['No', 'Yes']))

        return {"accuracy": acc, "precision": prec,
                "recall": rec, "f1": f1, "roc_auc": auc}


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════
class ModelEvaluator:
    """Produces evaluation plots for Naive Bayes on Play Tennis."""

    def __init__(self, model: NaiveBayesModel):
        self.model = model

    def evaluate(self, X, y_true, dataset_name="Dataset"):
        print(f"\n{'=' * 70}")
        print(f"MODEL EVALUATION — {dataset_name}")
        print(f"{'=' * 70}")

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['No', 'Yes']))

        self._plot_evaluation(y_true, y_pred, y_prob, dataset_name)
        self._plot_prediction_distribution(y_true, y_pred, y_prob, dataset_name)

    # ------------------------------------------------------------------
    def _plot_evaluation(self, y_true, y_pred, y_prob, dataset_name):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=DPI)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
        annot  = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
                            for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                    cbar_kws={"shrink": 0.8})
        axes[0, 0].set_title(f'Confusion Matrix — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0, 1].plot(fpr, tpr, lw=2, label=f'ROC (AUC={auc_val:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
        axes[0, 1].fill_between(fpr, tpr, alpha=0.2)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve — {dataset_name}', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Probability Distribution
        axes[1, 0].hist(y_prob[y_true == 0], bins=20, alpha=0.6,
                        label='No (Actual)', color='tomato', edgecolor='black')
        axes[1, 0].hist(y_prob[y_true == 1], bins=20, alpha=0.6,
                        label='Yes (Actual)', color='mediumseagreen', edgecolor='black')
        axes[1, 0].set_xlabel('Predicted Probability (Yes)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Probability Distribution — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Metrics Bar Chart
        names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        vals   = [accuracy_score(y_true, y_pred),
                  precision_score(y_true, y_pred, zero_division=0),
                  recall_score(y_true, y_pred, zero_division=0),
                  f1_score(y_true, y_pred, zero_division=0),
                  auc_val]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars   = axes[1, 1].bar(names, vals, color=colors, edgecolor='black', alpha=0.8)
        for b, v in zip(bars, vals):
            axes[1, 1].text(b.get_x() + b.get_width()/2, b.get_height(),
                            f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].set_title(f'Performance Metrics — {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fname = f'eval_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plot saved → {fname}")

    # ------------------------------------------------------------------
    def _plot_prediction_distribution(self, y_true, y_pred, y_prob, dataset_name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        # Confidence Analysis
        is_correct      = (y_true == y_pred)
        correct_probs   = y_prob[is_correct]
        incorrect_probs = y_prob[~is_correct]

        axes[0].hist(correct_probs,   bins=20, alpha=0.6,
                     label='Correct',   color='mediumseagreen', edgecolor='black')
        axes[0].hist(incorrect_probs, bins=20, alpha=0.6,
                     label='Incorrect', color='tomato',         edgecolor='black')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Confidence Analysis — {dataset_name}',
                          fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Calibration Plot
        n_bins     = 10
        bin_edges  = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mean_probs, mean_true = [], []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            if mask.sum() > 0:
                mean_probs.append(y_prob[mask].mean())
                mean_true.append(y_true[mask].mean())
            else:
                mean_probs.append(bin_centers[i])
                mean_true.append(0)

        axes[1].plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
        axes[1].plot(mean_probs, mean_true, 'o-', lw=2, ms=8,
                     label='Model Calibration', color='#1f77b4')
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Actual Positive Rate')
        axes[1].set_title(f'Calibration Plot — {dataset_name}',
                          fontsize=12, fontweight='bold')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'pred_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Prediction distribution saved → {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class MLPipeline:
    """End-to-end pipeline: Load → Validate → Visualize → Process → Train → Evaluate."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model     = None
        self.evaluator = None

    # ------------------------------------------------------------------
    def run(self):
        print("\n" + "=" * 70)
        print("NAIVE BAYES PIPELINE — PLAY TENNIS PREDICTION")
        print("=" * 70)

        # 1️⃣  Load
        data, target = self.loader.load_data()

        # 2️⃣  Validate
        validator = DatasetValidator(data, target)
        validator.verify_dataset()

        # 3️⃣  Visualize (raw categorical data)
        visualizer = TennisVisualizer(data, target)
        visualizer.visualize()

        # 4️⃣  Process (encode)
        self.processor = DatasetProcessor(data, target)
        X_enc, y_enc   = self.processor.process_dataset()

        # 5️⃣  Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc.values, y_enc.values,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_enc
        )
        print(f"\n✓ Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

        # 6️⃣  Train
        self.model = NaiveBayesModel(alpha=ALPHA)
        self.model.fit(X_train, y_train)

        # 7️⃣  Evaluate
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set")
        self.evaluator.evaluate(X_test,  y_test,  "Test Set")

        # 8️⃣  Cross-Validation
        self._perform_cross_validation(X_enc.values, y_enc.values)

        # 9️⃣  Predict a new sample
        self._predict_new_sample()

    # ------------------------------------------------------------------
    def _perform_cross_validation(self, X, y):
        print(f"\n{'=' * 70}")
        print("K-FOLD CROSS-VALIDATION (5-Fold)")
        print(f"{'=' * 70}")

        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        nb  = CategoricalNB(alpha=ALPHA)

        acc_scores  = cross_val_score(nb, X, y, cv=cv, scoring='accuracy')
        prec_scores = cross_val_score(nb, X, y, cv=cv, scoring='precision')
        rec_scores  = cross_val_score(nb, X, y, cv=cv, scoring='recall')
        f1_scores   = cross_val_score(nb, X, y, cv=cv, scoring='f1')
        auc_scores  = cross_val_score(nb, X, y, cv=cv, scoring='roc_auc')

        print(f"\nAccuracy  : {acc_scores.mean():.4f}  (+/- {acc_scores.std():.4f})")
        print(f"Precision : {prec_scores.mean():.4f}  (+/- {prec_scores.std():.4f})")
        print(f"Recall    : {rec_scores.mean():.4f}  (+/- {rec_scores.std():.4f})")
        print(f"F1-Score  : {f1_scores.mean():.4f}  (+/- {f1_scores.std():.4f})")
        print(f"ROC-AUC   : {auc_scores.mean():.4f}  (+/- {auc_scores.std():.4f})")

        verdict = 'GOOD generalization ✓' if acc_scores.std() < 0.05 else 'HIGH variance ⚠'
        print(f"\n✓ Cross-validation indicates {verdict}")

        # Plot cross-val scores
        self._plot_cross_validation(
            [acc_scores, prec_scores, rec_scores, f1_scores, auc_scores],
            ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        )

    # ------------------------------------------------------------------
    def _plot_cross_validation(self, score_lists, labels):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        # Box plot
        axes[0].boxplot(score_lists, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[0].set_title("CV Score Distribution (5-Fold)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis='y')

        # Mean ± std bar chart
        means = [s.mean() for s in score_lists]
        stds  = [s.std()  for s in score_lists]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[1].bar(labels, means, yerr=stds, capsize=5,
                           color=colors, edgecolor='black', alpha=0.8)
        for b, m in zip(bars, means):
            axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                         f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        axes[1].set_title("CV Mean ± Std", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1.15)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("nb_cross_validation.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Cross-validation plot saved")

    # ------------------------------------------------------------------
    def _predict_new_sample(self):
        print(f"\n{'=' * 70}")
        print("NEW SAMPLE PREDICTION")
        print(f"{'=' * 70}")

        # Raw new sample (same categories as training)
        new_raw = pd.DataFrame([{
            'outlook':  'Sunny',
            'temp':     'Cool',
            'humidity': 'Normal',
            'wind':     'Weak',
        }])

        # Encode using fitted label encoders
        new_enc = pd.DataFrame()
        for col in new_raw.columns:
            le  = self.processor.label_encoders[col]
            val = new_raw[col].astype(str)
            # Handle unseen labels gracefully
            known = set(le.classes_)
            val   = val.apply(lambda x: x if x in known else le.classes_[0])
            new_enc[col] = le.transform(val)

        prob = self.model.predict_proba(new_enc.values)[0]
        pred = self.model.predict(new_enc.values)[0]

        # Decode prediction
        pred_label = self.processor.target_encoder.inverse_transform([pred])[0]

        print(f"Input sample : {new_raw.iloc[0].to_dict()}")
        print(f"P(Yes)       : {prob:.4f}")
        print(f"P(No)        : {1 - prob:.4f}")
        print(f"Prediction   : {pred_label}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
