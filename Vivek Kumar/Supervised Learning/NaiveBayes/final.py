import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
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
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import joblib

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────────────────────────────────────
def ensure_nltk_resources():
    """Download required NLTK resources if not already present."""
    for resource in ['stopwords']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

ensure_nltk_resources()

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Dataset
RANDOM_STATE    = 42
DATASET_PATH    = "IMDB Dataset.csv"
TEST_SIZE       = 0.2
SAMPLE_SIZE     = 2000      # Set to None to use full dataset

# Vectorizer
VECTORIZER_TYPE = 'count'   # 'count' | 'tfidf'
MAX_FEATURES    = 10000     # Maximum vocabulary size
NGRAM_RANGE     = (1, 2)    # Unigrams + bigrams for richer features

# Model Persistence
MODEL_SAVE_PATH = "nb_sentiment_model.pkl"

# Cross-Validation
CV_FOLDS = 5

# Visualization
FIGURE_SIZE = (14, 10)
DPI         = 100
STYLE       = 'seaborn-v0_8-darkgrid'


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """Loads and performs initial preparation of the IMDB sentiment dataset."""

    def __init__(self, dataset_path: str = None, sample_size: Optional[int] = None):
        self.dataset_path = dataset_path
        self.sample_size  = sample_size
        self.data:   Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series]    = None

    # ------------------------------------------------------------------
    def _generate_synthetic_dataset(self, n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.Series]:
        """Create a tiny synthetic dataset so the pipeline runs without a CSV."""
        np.random.seed(RANDOM_STATE)

        positive_templates = [
            "this movie was absolutely wonderful and amazing",
            "great film loved every moment of it",
            "brilliant acting and superb direction highly recommend",
            "one of the best films I have ever seen fantastic",
            "outstanding performance by the entire cast truly impressive",
        ]
        negative_templates = [
            "this movie was terrible and a complete waste of time",
            "awful film boring and poorly written",
            "bad acting and dreadful direction avoid at all costs",
            "one of the worst films I have ever seen disappointing",
            "poor performance by the cast truly dreadful experience",
        ]

        n_each = n_samples // 2
        reviews   = (positive_templates * (n_each // len(positive_templates) + 1))[:n_each] + \
                    (negative_templates * (n_each // len(negative_templates) + 1))[:n_each]
        sentiments = ['positive'] * n_each + ['negative'] * n_each

        df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        self.data   = df[['review']]
        self.target = df['sentiment']
        return self.data, self.target

    # ------------------------------------------------------------------
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                print(f"✓ Loaded dataset from: {self.dataset_path}")

                if self.sample_size and self.sample_size < len(df):
                    df = df.sample(self.sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
                    print(f"✓ Sampled {self.sample_size} rows for faster training")

                self.data   = df[['review']]
                self.target = df['sentiment']
                print(f"✓ Samples : {len(self.data)}")
                print(f"✓ Columns : {list(df.columns)}")
                return self.data, self.target

            except Exception as ex:
                print(f"⚠ Failed to load {self.dataset_path}: {ex}")
                print("⚠ Falling back to synthetic dataset generation")

        self.data, self.target = self._generate_synthetic_dataset()
        print(f"✓ Synthetic IMDB dataset generated")
        print(f"✓ Samples: {len(self.data)}")
        return self.data, self.target


# ─────────────────────────────────────────────────────────────────────────────
# DATASET VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class DatasetValidator:
    """Validates the raw dataset before processing."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        passed = True

        # Empty check
        if self.data.empty or self.target.empty:
            print("✗ ERROR: Dataset is empty!")
            return False
        print("✓ Dataset is not empty")

        # Shape
        print(f"\n--- Dataset Shape ---")
        print(f"Reviews shape : {self.data.shape}")
        print(f"Target shape  : {self.target.shape}")

        if len(self.data) != len(self.target):
            print("✗ ERROR: Features and target row counts differ!")
            return False
        print("✓ Features and target have matching rows")

        # Missing values
        print(f"\n--- Missing Values ---")
        missing_reviews = self.data['review'].isnull().sum()
        missing_target  = self.target.isnull().sum()
        print(f"Missing reviews : {missing_reviews}")
        print(f"Missing targets : {missing_target}")
        if missing_reviews > 0 or missing_target > 0:
            print("⚠ WARNING: Dataset contains missing values")
            passed = False
        else:
            print("✓ No missing values detected")

        # Class distribution
        print(f"\n--- Class Distribution ---")
        counts = self.target.value_counts()
        print(counts)
        print(f"Class balance: {counts.min() / counts.max():.2f}")

        # Sample reviews
        print(f"\n--- Sample Reviews ---")
        print(self.data['review'].head(3).to_string())

        return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """Cleans, normalises, and stems raw review text."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.ps         = PorterStemmer()

    # ------------------------------------------------------------------
    @staticmethod
    def clean_html(text: str) -> str:
        """Strip HTML tags."""
        return re.sub(re.compile('<.*?>'), '', text)

    @staticmethod
    def remove_special(text: str) -> str:
        """Keep only alphanumeric characters and spaces."""
        return ''.join(ch for ch in text if ch.isalnum() or ch == ' ')

    # ------------------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline for a single review."""
        if not isinstance(text, str):
            text = str(text)
        text  = self.clean_html(text)
        text  = text.lower()
        text  = self.remove_special(text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        words = [self.ps.stem(w) for w in words]
        return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class DatasetProcessor:
    """Applies text preprocessing and vectorisation to produce numeric features."""

    def __init__(
        self,
        data:            pd.DataFrame,
        target:          pd.Series,
        vectorizer_type: str = VECTORIZER_TYPE,
        max_features:    int = MAX_FEATURES,
        ngram_range:     tuple = NGRAM_RANGE,
    ):
        self.data            = data.copy()
        self.target          = target.copy()
        self.vectorizer_type = vectorizer_type
        self.max_features    = max_features
        self.ngram_range     = ngram_range
        self.preprocessor    = TextPreprocessor()
        self.vectorizer      = None

    # ------------------------------------------------------------------
    def _build_vectorizer(self):
        params = dict(max_features=self.max_features, ngram_range=self.ngram_range)
        if self.vectorizer_type == 'tfidf':
            return TfidfVectorizer(**params)
        return CountVectorizer(**params)

    # ------------------------------------------------------------------
    def process_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        # ── 1. Handle missing values ──────────────────────────────────
        print("\n--- Handling Missing Values ---")
        before = self.data['review'].isnull().sum()
        self.data['review'].fillna('', inplace=True)
        self.target.fillna(self.target.mode()[0], inplace=True)
        print(f"Filled {before} missing reviews with empty string")

        # ── 2. Text Preprocessing ─────────────────────────────────────
        print("\n--- Text Preprocessing (Clean → Lower → Remove specials → Stopwords → Stem) ---")
        self.data['review'] = self.data['review'].apply(self.preprocessor.preprocess)
        print("✓ Text preprocessing complete")

        # ── 3. Label Encoding ─────────────────────────────────────────
        print("\n--- Label Encoding ---")
        label_map = {'positive': 1, 'negative': 0}
        if self.target.dtype == object:
            processed_target = self.target.map(label_map)
            if processed_target.isnull().any():
                # Fallback: use median split
                print("⚠ Unknown labels detected — using median split")
                numeric = pd.to_numeric(self.target, errors='coerce')
                processed_target = (numeric > numeric.median()).astype(int)
        else:
            processed_target = self.target.astype(int)
        print(f"Target classes: {sorted(processed_target.unique())}")

        # ── 4. Vectorisation ──────────────────────────────────────────
        # ─── NB-SPECIFIC NOTE ─────────────────────────────────────────
        # Naive Bayes works on raw counts / TF-IDF weights (non-negative).
        # Unlike KNN, we do NOT z-score standardise features here because
        # (a) negative values break MultinomialNB, and
        # (b) NB's conditional independence assumption is scale-invariant.
        # ──────────────────────────────────────────────────────────────
        print(f"\n--- Vectorisation ({self.vectorizer_type.upper()}, max_features={self.max_features}) ---")
        print("  ⓘ  Naive Bayes requires non-negative feature values.")
        print("     Using word counts / TF-IDF (no z-score scaling).")
        self.vectorizer    = self._build_vectorizer()
        processed_features = self.vectorizer.fit_transform(self.data['review']).toarray()
        print(f"✓ Feature matrix shape: {processed_features.shape}")

        return processed_features, processed_target.values


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISER
# ─────────────────────────────────────────────────────────────────────────────

class SentimentVisualizer:
    """Generates EDA and post-training visualisations for the IMDB dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target
        plt.style.use(STYLE)

    # ------------------------------------------------------------------
    def visualize(self):
        print(f"\n{'=' * 70}")
        print("SENTIMENT DATASET VISUALISATION")
        print(f"{'=' * 70}")

        self.plot_target_distribution()
        self.plot_review_length_distribution()
        self.plot_top_words()
        self.plot_word_length_boxplot()
        self.plot_review_length_kde()

        print("✓ All visualisations saved")

    # ── 1. Class Distribution ─────────────────────────────────────────
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_index()
        labels = [str(v) for v in counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        bars = axes[0].bar(labels, counts.values,
                           color=['green', 'red'], edgecolor='black', alpha=0.7)
        axes[0].set_title("Sentiment Class Distribution", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3, axis='y')
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h,
                         f'{int(h)}', ha='center', va='bottom')

        axes[1].pie(counts.values, labels=['Negative', 'Positive'],
                    autopct='%1.1f%%', colors=['red', 'green'],
                    startangle=90, explode=(0.05, 0.05))
        axes[1].set_title("Class Ratio", fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig("sentiment_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Target distribution plot saved")

    # ── 2. Review Length Distribution ────────────────────────────────
    def plot_review_length_distribution(self):
        lengths = self.data['review'].apply(lambda x: len(str(x).split()))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
        for label, color in zip([0, 1], ['red', 'green']):
            mask = self.target == label
            lbl  = 'Negative' if label == 0 else 'Positive'
            arr = lengths[mask]
            if arr.empty or arr.sum() == 0:
                print(f"[SKIP] No data for {lbl} reviews in review length distribution.")
                continue
            axes[0].hist(arr, bins=40, alpha=0.6,
                         label=lbl, color=color, edgecolor='black')
        axes[0].set_title("Review Length Distribution by Sentiment", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        box_data = [lengths[self.target == 0], lengths[self.target == 1]]
        box_data = [arr for arr in box_data if not arr.empty and arr.sum() > 0]
        if box_data:
            axes[1].boxplot(
                box_data,
                labels=['Negative', 'Positive'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
            )
            axes[1].set_title("Review Length Boxplot", fontsize=12, fontweight='bold')
            axes[1].set_ylabel("Word Count")
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1].text(0.5, 0.5, 'No data for boxplot', ha='center', va='center')
        plt.tight_layout()
        plt.savefig("sentiment_review_length.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Review length distribution saved")

    # ── 3. Top Words per Class ────────────────────────────────────────
    def plot_top_words(self, top_n: int = 20):
        ps = PorterStemmer()
        sw = set(stopwords.words('english'))

        def get_top_words(texts):
            from collections import Counter
            words = []
            for t in texts:
                for w in str(t).lower().split():
                    clean = re.sub(r'[^a-z]', '', w)
                    if clean and clean not in sw:
                        words.append(ps.stem(clean))
            return Counter(words).most_common(top_n)

        pos_words = get_top_words(self.data['review'][self.target == 1])
        neg_words = get_top_words(self.data['review'][self.target == 0])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=DPI)

        for ax, words, color, title in zip(
            axes,
            [pos_words, neg_words],
            ['green', 'red'],
            ['Top Words — Positive Reviews', 'Top Words — Negative Reviews']
        ):
            if words:
                w, c = zip(*words)
                ax.barh(list(w)[::-1], list(c)[::-1], color=color, alpha=0.7, edgecolor='black')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("Frequency")
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig("sentiment_top_words.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Top words plot saved")

    # ── 4. Word Length Boxplot ────────────────────────────────────────
    def plot_word_length_boxplot(self):
        avg_word_len = self.data['review'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
        )

        plt.figure(figsize=(8, 5), dpi=DPI)
        plt.boxplot(
            [avg_word_len[self.target == 0], avg_word_len[self.target == 1]],
            labels=['Negative', 'Positive'],
            patch_artist=True,
            boxprops=dict(facecolor='lightcoral'),
        )
        plt.title("Average Word Length by Sentiment", fontsize=12, fontweight='bold')
        plt.ylabel("Avg Word Length (chars)")
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("sentiment_word_length_boxplot.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Word length boxplot saved")

    # ── 5. Review Length KDE ──────────────────────────────────────────
    def plot_review_length_kde(self):
        lengths = self.data['review'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 5), dpi=DPI)
        for label, color, lbl in zip([1, 0], ['green', 'red'], ['Positive', 'Negative']):
            arr = lengths[self.target == label]
            if arr.empty or arr.sum() == 0:
                print(f"[SKIP] No data for {lbl} reviews in KDE plot.")
                continue
            arr.plot.kde(linewidth=2, label=lbl, color=color)
        plt.title("Review Length KDE by Sentiment", fontsize=12, fontweight='bold')
        plt.xlabel("Word Count")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("sentiment_review_length_kde.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Review length KDE saved")


# ─────────────────────────────────────────────────────────────────────────────
# NAIVE BAYES MODEL
# ─────────────────────────────────────────────────────────────────────────────

class NaiveBayesModel:
    """
    Naive Bayes sentiment classifier.

    Trains both MultinomialNB and BernoulliNB variants and exposes a
    unified predict / evaluate interface keyed by model name.
    """

    VARIANTS = {
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB":   BernoulliNB(),
    }

    def __init__(self):
        self.models: Dict[str, object] = {k: v for k, v in self.VARIANTS.items()}
        self.best_model_name: Optional[str] = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"\n{'=' * 60}")
        print("TRAINING NAIVE BAYES MODELS")
        print(f"{'=' * 60}")

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"✓ {name} trained successfully")

        try:
            joblib.dump(self.models, MODEL_SAVE_PATH)
            print(f"✓ Models saved to {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"⚠ Could not save models: {ex}")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, model_name: str = "MultinomialNB") -> np.ndarray:
        return self.models[model_name].predict(X)

    def predict_proba(self, X: np.ndarray, model_name: str = "MultinomialNB") -> np.ndarray:
        proba = self.models[model_name].predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()

    # ------------------------------------------------------------------
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, dict]:
        """Evaluate every variant and return their metrics."""
        results = {}
        for name in self.models:
            print(f"\n{'=' * 60}")
            print(f"EVALUATION — {name}")
            print(f"{'=' * 60}")
            y_pred = self.predict(X_test, name)
            y_prob = self.predict_proba(X_test, name)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_prob)

            print(f"Accuracy : {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall   : {rec:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print(f"ROC-AUC  : {auc:.4f}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred,
                                        target_names=['Negative', 'Positive']))

            results[name] = dict(accuracy=acc, precision=prec,
                                 recall=rec, f1=f1, roc_auc=auc)

        # Identify best model by F1
        self.best_model_name = max(results, key=lambda k: results[k]['f1'])
        print(f"\n★ Best model by F1: {self.best_model_name} "
              f"(F1={results[self.best_model_name]['f1']:.4f})")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """Generates comprehensive evaluation plots for each NB variant."""

    def __init__(self, model: NaiveBayesModel):
        self.model = model

    # ------------------------------------------------------------------
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, dataset_name: str = "Dataset"):
        for model_name in self.model.models:
            y_pred = self.model.predict(X, model_name)
            y_prob = self.model.predict_proba(X, model_name)
            self._plot_evaluation(y_true, y_pred, y_prob, dataset_name, model_name)
            self._plot_prediction_analysis(y_true, y_pred, y_prob, dataset_name, model_name)

    # ------------------------------------------------------------------
    def _plot_evaluation(self, y_true, y_pred, y_prob, dataset_name, model_name):
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, dpi=DPI)
        tag = f"{model_name} — {dataset_name}"

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
        annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
                           for j in range(cm.shape[1])]
                          for i in range(cm.shape[0])])
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={"shrink": 0.8})
        axes[0, 0].set_title(f'Confusion Matrix\n{tag}', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_val:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        axes[0, 1].fill_between(fpr, tpr, alpha=0.2)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve\n{tag}', fontsize=11, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Probability Distribution
        axes[1, 0].hist(y_prob[y_true == 0], bins=25, alpha=0.6,
                        label='Negative (actual)', color='red', edgecolor='black')
        axes[1, 0].hist(y_prob[y_true == 1], bins=25, alpha=0.6,
                        label='Positive (actual)', color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Predicted Probability (Positive)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Probability Distribution\n{tag}', fontsize=11, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Metrics Bar Chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values  = [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
            auc_val,
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[1, 1].bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_title(f'Performance Metrics\n{tag}', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        fname = f'evaluation_{model_name.lower()}_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plots saved → {fname}")

    # ------------------------------------------------------------------
    def _plot_prediction_analysis(self, y_true, y_pred, y_prob, dataset_name, model_name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
        tag = f"{model_name} — {dataset_name}"

        # Confidence analysis
        correct   = y_pred == y_true
        axes[0].hist(y_prob[correct], bins=25, alpha=0.6,
                     label='Correct', color='green', edgecolor='black')
        axes[0].hist(y_prob[~correct], bins=25, alpha=0.6,
                     label='Incorrect', color='red', edgecolor='black')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Prediction Confidence Analysis\n{tag}', fontsize=11, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Calibration plot
        n_bins    = 10
        edges     = np.linspace(0, 1, n_bins + 1)
        mean_prob, mean_true = [], []
        for i in range(n_bins):
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
            if mask.sum() > 0:
                mean_prob.append(y_prob[mask].mean())
                mean_true.append(y_true[mask].mean())
            else:
                mean_prob.append((edges[i] + edges[i + 1]) / 2)
                mean_true.append(0)

        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        axes[1].plot(mean_prob, mean_true, 'o-', linewidth=2, markersize=8,
                     label='Model', color='#1f77b4')
        axes[1].fill_between(mean_prob, mean_true,
                             np.linspace(0, 1, len(mean_prob)), alpha=0.2)
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Actual Positive Rate')
        axes[1].set_title(f'Calibration Plot\n{tag}', fontsize=11, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'predictions_{model_name.lower()}_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Prediction analysis saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# ML PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class MLPipeline:
    """End-to-end pipeline for IMDB Sentiment Analysis using Naive Bayes."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH, sample_size=SAMPLE_SIZE)
        self.processor = None
        self.model     = None
        self.evaluator = None

    # ------------------------------------------------------------------
    def run(self):
        print(f"\n{'=' * 70}")
        print("NAIVE BAYES PIPELINE — SENTIMENT ANALYSIS (IMDB)")
        print(f"{'=' * 70}")

        # 1️⃣  Load
        data, target = self.loader.load_data()

        # 2️⃣  Validate
        validator = DatasetValidator(data, target)
        validator.verify_dataset()

        # 3️⃣  Visualise (raw data)
        visualizer = SentimentVisualizer(data, target)
        visualizer.visualize()

        # 4️⃣  Process (preprocess + vectorise)
        self.processor = DatasetProcessor(data, target)
        X, y = self.processor.process_dataset()

        # 5️⃣  Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        print(f"\n✓ Train size : {X_train.shape[0]}")
        print(f"✓ Test size  : {X_test.shape[0]}")

        # 6️⃣  Train
        self.model = NaiveBayesModel()
        self.model.fit(X_train, y_train)

        # 7️⃣  Evaluate both variants
        results = self.model.evaluate_all(X_test, y_test)

        # 8️⃣  Generate full evaluation plots (train + test)
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set")
        self.evaluator.evaluate(X_test,  y_test,  "Test Set")

        # 9️⃣  Cross-Validation
        self._perform_cross_validation(X, y)

        # 🔟  Live prediction demo
        self._demo_prediction()

    # ------------------------------------------------------------------
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray):
        print(f"\n{'=' * 70}")
        print(f"K-FOLD CROSS-VALIDATION ({CV_FOLDS}-Fold) — MultinomialNB")
        print(f"{'=' * 70}")

        cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        clf = MultinomialNB()

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(clf, X, y, cv=cv, scoring=metric)
            print(f"{metric.capitalize():10s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        generalisation = "GOOD ✓" if acc_scores.std() < 0.03 else "HIGH VARIANCE ⚠"
        print(f"\n✓ Cross-validation indicates {generalisation}")

    # ------------------------------------------------------------------
    def _demo_prediction(self):
        """Run the best model on a few hand-crafted reviews."""
        print(f"\n{'=' * 70}")
        print("LIVE PREDICTION DEMO")
        print(f"{'=' * 70}")

        test_reviews = [
            "This movie was absolutely fantastic! The acting was superb and the plot was gripping.",
            "Terrible film. Boring, predictable, and a complete waste of two hours.",
            "It was okay. Some parts were good but overall a mediocre experience.",
        ]
        expected = ["Positive", "Negative", "Neutral/Mixed"]

        best = self.model.best_model_name or "MultinomialNB"
        pre  = self.processor.preprocessor

        for review, exp in zip(test_reviews, expected):
            cleaned    = pre.preprocess(review)
            vectorized = self.processor.vectorizer.transform([cleaned]).toarray()
            prob       = self.model.predict_proba(vectorized, best)[0]
            pred_label = "Positive" if prob >= 0.5 else "Negative"

            print(f"\nReview  : {review[:80]}...")
            print(f"Expected: {exp}")
            print(f"P(pos)  : {prob:.4f}  →  Predicted: {pred_label}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the Naive Bayes sentiment analysis pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n❌ ERROR during pipeline execution:")
        print(f"{type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    main()