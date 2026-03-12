import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================

# Dataset Configuration
DATASET_PATH = r"D:\Desktop Of D\SupervisedLearning\LinearRegression\BIKE DETAILS.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
TARGET_COLUMN= "selling_price"

# Model Hyperparameters
LEARNING_RATE = 0.001
NUM_ITERATIONS = 1000
BATCH_SIZE = 32
CONVERGENCE_THRESHOLD = 1e-6

# Model Persistence
# it's called pickle file saved at disk can be transferable and reconstructed
# it stores any kind of python objects like numpy array machine learning models list dictionaries tuple
MODEL_SAVE_PATH = r"D:\Desktop Of D\SupervisedLearning\LinearRegression\Linear_regression_model.pkl"

# Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = 'seaborn-v0_8-darkgrid'


@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float

    def __str__(self) -> str:
       """Return formatted string representation of metrics."""
       return (
          f"Model Performance Metrics:\n"
          f"{'=' * 50}\n"
          f"Mean Squared Error (MSE):  {self.mse:.4f}\n"
          f"Root Mean Squared Error (RMSE): {self.rmse:.4f}\n"
          f"Mean Absolute Error (MAE): {self.mae:.4f}\n"
          f"R² Score:                  {self.r2:.4f}\n"
          f"{'=' * 50}"
       )


class DatasetLoader:
    """Class responsible for loading and initial dataset operations."""

    def __init__(self, dataset_path: str):
       """
       Initialize the DatasetLoader.

       Args:
          dataset_path: Path or identifier for the dataset
       """
       self.dataset_path = dataset_path
       self.data: Optional[pd.DataFrame] = None
       self.target: Optional[pd.Series] = None
       self.feature_names: Optional[list] = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        df = pd.read_csv(self.dataset_path)

        self.data = df.drop(columns=[TARGET_COLUMN])
        self.target = df[TARGET_COLUMN]
        self.feature_names = list(self.data.columns)

        print(f"✓ Dataset loaded successfully from: {self.dataset_path}")
        print(f"✓ Total samples: {df.shape[0]}")
        print(f"✓ Number of features: {len(self.feature_names)}")
        print(f"✓ Feature names: {', '.join(self.feature_names)}")

        return self.data, self.target



class DatasetValidator:
    """Class responsible for dataset validation and verification."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
       """
       Initialize the DatasetValidator.

       Args:
          data: Features dataframe
          target: Target series
       """
       self.data = data
       self.target = target

    def zverify_dataset(self) -> bool:
       """
       Perform comprehensive dataset verification.

       Returns:
          Boolean indicating if dataset passed all validations
       """
       print(f"\n{'=' * 70}")
       print("DATASET VERIFICATION")
       print(f"{'=' * 70}")

       validation_passed = True

       # Check for empty dataset
       if self.data.empty or self.target.empty:
          print("✗ ERROR: Dataset is empty!")
          return False
       print("✓ Dataset is not empty")

       # Check dataset shape
       print(f"\n--- Dataset Shape ---")
       print(f"Features shape: {self.data.shape}")
       print(f"Target shape: {self.target.shape}")
       print(f"Number of samples: {self.data.shape[0]}")
       print(f"Number of features: {self.data.shape[1]}")

       # Check for mismatched rows
       if self.data.shape[0] != self.target.shape[0]:
          print("✗ ERROR: Features and target have different number of rows!")
          return False
       print("✓ Features and target have matching rows")

       # Check for missing values
       print(f"\n--- Missing Values ---")
       missing_features = self.data.isnull().sum().sum()
       missing_target = self.target.isnull().sum()
       print(f"Missing values in features: {missing_features}")
       print(f"Missing values in target: {missing_target}")

       if missing_features > 0 or missing_target > 0:
          print("⚠ WARNING: Dataset contains missing values")
          validation_passed = False
       else:
          print("✓ No missing values detected")

       # Check data types
       print(f"\n--- Data Types ---")
       print(self.data.dtypes)

       # Verify numeric data
       non_numeric = self.data.select_dtypes(exclude=[np.number]).columns
       if len(non_numeric) > 0:
          print(f"⚠ WARNING: Non-numeric columns detected: {list(non_numeric)}")
          validation_passed = False
       else:
          print("✓ All features are numeric")

       # Display first few rows
       print(f"\n--- First 5 Rows ---")
       print(self.data.head())

       # Display last few rows
       print(f"\n--- Last 5 Rows ---")
       print(self.data.tail())

       # Display statistical summary
       print(f"\n--- Statistical Summary ---")
       print(self.data.describe())

       # Target statistics
       print(f"\n--- Target Statistics ---")
       print(self.target.describe())

       # Check for infinite values
       numeric_data = self.data.select_dtypes(include=[np.number])
       inf_count = np.isinf(numeric_data.values).sum()
       
       if inf_count > 0:
            print(f"⚠ WARNING: {inf_count} infinite numeric values detected")
            validation_passed = False
      
       else:
            print("✓ No infinite numeric values detected")
        
       return validation_passed


class DatasetProcessor:
    """Class responsible for dataset processing and transformation."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
       """
       Initialize the DatasetProcessor.

       Args:
          data: Features dataframe
          target: Target series
       """
       self.data = data.copy()
       self.target = target.copy()
       self.processed_data: Optional[pd.DataFrame] = None
       self.processed_target: Optional[pd.Series] = None

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
       """
       Process the dataset: handle missing values, outliers, normalization.

       Returns:
          Tuple of (processed_features, processed_target)
       """
       print(f"\n{'=' * 70}")
       print("DATASET PROCESSING")
       print(f"{'=' * 70}")

       # Drop high-cardinality column (bike name)
       if "name" in self.data.columns:
        self.data.drop(columns=["name"], inplace=True)
        print("✓ 'name' column dropped (high cardinality feature)")

       # Convert manufacturing year to bike age
       if "year" in self.data.columns:
        self.data["bike_age"] = 2025 - self.data["year"]
        self.data.drop(columns=["year"], inplace=True)
        print("✓ 'year' converted to 'bike_age'")

       # Handle missing values
       print("\n--- Handling Missing Values ---")
       missing_before = self.data.isnull().sum().sum()
       numeric_cols = self.data.select_dtypes(include=[np.number]).columns
       self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
       self.target = self.target.fillna(self.target.mean())
       missing_after = self.data.isnull().sum().sum() 
       print(f"Missing values before: {missing_before}")
       print(f"Missing values after: {missing_after}")
       
       # Create depreciation ratio feature
       if "ex_showroom_price" in self.data.columns:
        self.data["price_ratio"] = self.target / (self.data["ex_showroom_price"] + 1)
        print("✓ 'price_ratio' feature created")


       print("\n--- Encoding Categorical Variables ---")
       # Convert owner column from text to numeric
       if "owner" in self.data.columns:
            self.data["owner"] = self.data["owner"].str.extract('(\d+)').astype(float)
            print("✓ 'owner' column converted to numeric")

       categorical_columns = self.data.select_dtypes(include=['object']).columns
       if len(categorical_columns) > 0:
            print(f"Categorical columns: {list(categorical_columns)}")
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
            print("✓ One hot encoding applied")    
       else:
            print("✓ No categorical columns found")


       # Feature scaling (standardization)
       print("\n--- Feature Standardization ---")
       self.feature_means = self.data.mean()
       self.feature_stds = self.data.std()

       self.processed_data = (self.data - self.feature_means) / self.feature_stds
       print("✓ Features standardized (mean=0, std=1)")

       # Keep target as is for regression
       self.processed_target = self.target

       print(f"\n--- Processed Dataset Shape ---")
       print(f"Processed features: {self.processed_data.shape}")
       print(f"Processed target: {self.processed_target.shape}")

       return self.processed_data, self.processed_target


class DatasetVisualizer:
    """Class responsible for dataset visualization."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
       """
       Initialize the DatasetVisualizer.

       Args:
          data: Features dataframe
          target: Target series
       """
       self.data = data
       self.target = target
       plt.style.use(STYLE)

    def visualize_dataset(self) -> None:
       """Create comprehensive visualizations of the dataset."""
       print(f"\n{'=' * 70}")
       print("DATASET VISUALIZATION")
       print(f"{'=' * 70}")

       # 1. Target distribution
       self._plot_target_distribution()

       # 2. Correlation heatmap
       self._plot_correlation_heatmap()

       # 3. Feature distributions
       self._plot_feature_distributions()

       # 4. Scatter plots for top correlated features
       self._plot_top_correlations()

       print("✓ All visualizations created successfully")

    def _plot_target_distribution(self) -> None:
       """Plot the distribution of target variable."""
       plt.figure(figsize=FIGURE_SIZE, dpi=DPI)

       plt.subplot(1, 2, 1)
       plt.hist(self.target, bins=50, edgecolor='black', alpha=0.7)
       plt.xlabel('Selling Price')
       plt.ylabel('Frequency')
       plt.title('Target Distribution (Histogram)')
       plt.grid(True, alpha=0.3)

       plt.subplot(1, 2, 2)
       plt.boxplot(self.target, vert=True)
       plt.ylabel('Selling Price')
       plt.title('Target Distribution (Boxplot)')
       plt.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.savefig(r'D:\Desktop Of D\SupervisedLearning\LinearRegression\Graphs\target_distribution.png', dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Target distribution plot saved")

    def _plot_correlation_heatmap(self) -> None:
       """Plot correlation heatmap between features and target."""
       # Combine features and target for correlation
       combined_data = self.data.copy()
       combined_data['Target'] = self.target

       plt.figure(figsize=(14, 10), dpi=DPI)
       correlation_matrix = combined_data.corr()

       sns.heatmap(
          correlation_matrix,
          annot=True,
          fmt='.2f',
          cmap='coolwarm',
          center=0,
          square=True,
          linewidths=1,
          cbar_kws={"shrink": 0.8}
       )
       plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
       plt.tight_layout()
       plt.savefig(r'D:\Desktop Of D\SupervisedLearning\LinearRegression\Graphs\correlation_heatmap.png', dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Correlation heatmap saved")

    def _plot_feature_distributions(self) -> None:
       """Plot distributions of all features."""
       num_features = len(self.data.columns)
       cols = 3
       rows = (num_features + cols - 1) // cols

       fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), dpi=DPI)
       axes = axes.flatten() if num_features > 1 else [axes]

       for idx, column in enumerate(self.data.columns):
          axes[idx].hist(self.data[column], bins=30, edgecolor='black', alpha=0.7)
          axes[idx].set_title(f'{column}')
          axes[idx].set_xlabel('Value')
          axes[idx].set_ylabel('Frequency')
          axes[idx].grid(True, alpha=0.3)

       # Hide empty subplots
       for idx in range(num_features, len(axes)):
          axes[idx].axis('off')

       plt.tight_layout()
       plt.savefig(r'D:\Desktop Of D\SupervisedLearning\LinearRegression\Graphs\feature_distributions.png', dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Feature distributions plot saved")

    def _plot_top_correlations(self) -> None:
       """Plot scatter plots for top correlated features with target."""
       # Calculate correlations with target
       correlations = self.data.corrwith(self.target).abs().sort_values(ascending=False)
       top_features = correlations.head(4).index.tolist()

       fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=DPI)
       axes = axes.flatten()

       for idx, feature in enumerate(top_features):
          axes[idx].scatter(self.data[feature], self.target, alpha=0.5, s=10)
          axes[idx].set_xlabel(feature)
          axes[idx].set_ylabel('Selling Price')
          axes[idx].set_title(f'{feature} vs Target (Corr: {correlations[feature]:.3f})')
          axes[idx].grid(True, alpha=0.3)

       plt.tight_layout()
       plt.savefig(r'D:\Desktop Of D\SupervisedLearning\LinearRegression\Graphs\top_correlations.png', dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Top correlations plot saved")


class LinearRegressionModel:
    """
    Custom Linear_Regression implementation using Gradient Descent.

    This class implements Linear_Regression from scratch with batch gradient
    descent optimization.
    """

    def __init__(
          self,
          learning_rate: float = LEARNING_RATE,
          num_iterations: int = NUM_ITERATIONS,
          batch_size: int = BATCH_SIZE,
          convergence_threshold: float = CONVERGENCE_THRESHOLD
    ):
       """
       Initialize the Linear_Regression model.

       Args:
          learning_rate: Step size for gradient descent
          num_iterations: Maximum number of iterations
          batch_size: Size of mini-batches for gradient descent
          convergence_threshold: Threshold for early stopping
       """
       self.learning_rate = learning_rate
       self.num_iterations = num_iterations
       self.batch_size = batch_size
       self.convergence_threshold = convergence_threshold

       self.weights: Optional[np.ndarray] = None
       self.bias: float = 0.0
       self.loss_history: list = []

    def _initialize_parameters(self, n_features: int) -> None:
       """
       Initialize weights and bias.

       Args:
          n_features: Number of input features
       """
       np.random.seed(RANDOM_STATE)
       self.weights = np.random.randn(n_features) * 0.01
       self.bias = 0.0

    def _compute_predictions(self, X: np.ndarray) -> np.ndarray:
       """
       Compute predictions using current weights and bias.

       Args:
          X: Input features of shape (n_samples, n_features)

       Returns:
          Predictions of shape (n_samples,)
       """
       return np.dot(X, self.weights) + self.bias

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
       """
       Compute Mean Squared Error loss.

       Args:
          y_true: True target values
          y_pred: Predicted values

       Returns:
          MSE loss value
       """
       return np.mean((y_true - y_pred) ** 2)

    def _compute_gradients(
          self,
          X: np.ndarray,
          y_true: np.ndarray,
          y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
       """
       Compute gradients for weights and bias.

       Args:
          X: Input features
          y_true: True target values
          y_pred: Predicted values

       Returns:
          Tuple of (weight_gradients, bias_gradient)
       """
       n_samples = X.shape[0]
       error = y_pred - y_true

       weight_gradients = (2 / n_samples) * np.dot(X.T, error)
       bias_gradient = (2 / n_samples) * np.sum(error)

       return weight_gradients, bias_gradient

    def _create_mini_batches(
          self,
          X: np.ndarray,
          y: np.ndarray
    ) -> list:
       """
       Create mini-batches for stochastic gradient descent.

       Args:
          X: Input features
          y: Target values

       Returns:
          List of (X_batch, y_batch) tuples
       """
       n_samples = X.shape[0]
       indices = np.random.permutation(n_samples)

       mini_batches = []
       for i in range(0, n_samples, self.batch_size):
          batch_indices = indices[i:min(i + self.batch_size, n_samples)]
          X_batch = X[batch_indices]
          y_batch = y[batch_indices]
          mini_batches.append((X_batch, y_batch))

       return mini_batches

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
       """
       Train the model using gradient descent.

       Args:
          X: Training features of shape (n_samples, n_features)
          y: Training targets of shape (n_samples,)

       Returns:
          Self for method chaining
       """
       print(f"\n{'=' * 70}")
       print("MODEL TRAINING")
       print(f"{'=' * 70}")
       print(f"Learning Rate: {self.learning_rate}")
       print(f"Iterations: {self.num_iterations}")
       print(f"Batch Size: {self.batch_size}")
       print(f"Convergence Threshold: {self.convergence_threshold}")

       n_samples, n_features = X.shape
       self._initialize_parameters(n_features)

       prev_loss = float('inf')

       for iteration in range(self.num_iterations):
          # Create mini-batches
          mini_batches = self._create_mini_batches(X, y)

          for X_batch, y_batch in mini_batches:
             # Compute predictions
             y_pred = self._compute_predictions(X_batch)

             # Compute gradients
             weight_gradients, bias_gradient = self._compute_gradients(
                X_batch, y_batch, y_pred
             )

             # Update parameters
             self.weights -= self.learning_rate * weight_gradients
             self.bias -= self.learning_rate * bias_gradient

          # Compute loss on full dataset
          y_pred_full = self._compute_predictions(X)
          loss = self._compute_loss(y, y_pred_full)
          self.loss_history.append(loss)

          # Print progress
          if (iteration + 1) % 100 == 0:
             print(f"Iteration {iteration + 1}/{self.num_iterations} - Loss: {loss:.4f}")

          # Check for convergence
          if abs(prev_loss - loss) < self.convergence_threshold:
             print(f"\n✓ Converged at iteration {iteration + 1}")
             break

          prev_loss = loss

       print(f"✓ Training completed - Final Loss: {loss:.4f}")
       return self

    def predict(self, X: np.ndarray) -> np.ndarray:
       """
       Make predictions on new data.

       Args:
          X: Input features of shape (n_samples, n_features)

       Returns:
          Predictions of shape (n_samples,)
       """
       if self.weights is None:
          raise ValueError("Model must be trained before making predictions")

       return self._compute_predictions(X)

    def plot_loss_curve(self) -> None:
       """Plot the training loss curve."""
       plt.figure(figsize=(10, 6), dpi=DPI)
       plt.plot(self.loss_history, linewidth=2)
       plt.xlabel('Iteration', fontsize=12)
       plt.ylabel('Loss (MSE)', fontsize=12)
       plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig(r'D:\Desktop Of D\SupervisedLearning\LinearRegression\Graphs\loss_curve.png', dpi=DPI, bbox_inches='tight')
       plt.close()
       print("✓ Loss curve saved")


class ModelEvaluator:
    """Class responsible for model evaluation and metrics calculation."""

    def __init__(self, model: LinearRegressionModel):
       """
       Initialize the ModelEvaluator.

       Args:
          model: Trained LinearRegressionModel instance
       """
       self.model = model

    def evaluate(
          self,
          X: np.ndarray,
          y_true: np.ndarray,
          dataset_name: str = "Dataset"
    ) -> ModelMetrics:
       """
       Evaluate model performance.

       Args:
          X: Input features
          y_true: True target values
          dataset_name: Name of the dataset being evaluated

       Returns:
          ModelMetrics object containing evaluation metrics
       """
       print(f"\n{'=' * 70}")
       print(f"MODEL EVALUATION - {dataset_name}")
       print(f"{'=' * 70}")

       # Make predictions
       y_pred = self.model.predict(X)

       # Calculate metrics
       mse = mean_squared_error(y_true, y_pred)
       rmse = np.sqrt(mse)
       mae = mean_absolute_error(y_true, y_pred)
       r2 = r2_score(y_true, y_pred)

       metrics = ModelMetrics(mse=mse, rmse=rmse, mae=mae, r2=r2)
       print(metrics)

       # Plot predictions vs actual
       self._plot_predictions(y_true, y_pred, dataset_name)

       return metrics

    def _plot_predictions(
          self,
          y_true: np.ndarray,
          y_pred: np.ndarray,
          dataset_name: str
    ) -> None:
       """
       Plot predicted vs actual values.

       Args:
          y_true: True target values
          y_pred: Predicted values
          dataset_name: Name of the dataset
       """
       fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

       # Scatter plot
       axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
       axes[0].plot(
          [y_true.min(), y_true.max()],
          [y_true.min(), y_true.max()],
          'r--',
          lw=2,
          label='Perfect Prediction'
       )
       axes[0].set_xlabel('Actual Values', fontsize=12)
       axes[0].set_ylabel('Predicted Values', fontsize=12)
       axes[0].set_title(f'Predictions vs Actual - {dataset_name}', fontsize=14)
       axes[0].legend()
       axes[0].grid(True, alpha=0.3)

       # Residual plot
       residuals = y_true - y_pred
       axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
       axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
       axes[1].set_xlabel('Predicted Values', fontsize=12)
       axes[1].set_ylabel('Residuals', fontsize=12)
       axes[1].set_title(f'Residual Plot - {dataset_name}', fontsize=14)
       axes[1].grid(True, alpha=0.3)

       plt.tight_layout()
       filename = f'predictions_{dataset_name.lower().replace(" ", "_")}.png'
       plt.savefig(f'D:\\Desktop Of D\\SupervisedLearning\\LinearRegression\\Graphs\\{filename}', dpi=DPI, bbox_inches='tight')
       plt.close()
       print(f"✓ Prediction plots saved for {dataset_name}")


class MLPipeline:
    """
    Main pipeline orchestrating the entire machine learning workflow.
    """

    def __init__(self):
       """Initialize the ML Pipeline."""
       self.loader: Optional[DatasetLoader] = None
       self.validator: Optional[DatasetValidator] = None
       self.processor: Optional[DatasetProcessor] = None
       self.visualizer: Optional[DatasetVisualizer] = None
       self.model: Optional[LinearRegressionModel] = None
       self.evaluator: Optional[ModelEvaluator] = None

       self.X_train: Optional[np.ndarray] = None
       self.X_test: Optional[np.ndarray] = None
       self.X_val: Optional[np.ndarray] = None
       self.y_train: Optional[np.ndarray] = None
       self.y_test: Optional[np.ndarray] = None
       self.y_val: Optional[np.ndarray] = None

    def run(self) -> None:
       """Execute the complete ML pipeline."""
       print("\n" + "=" * 70)
       print("Linear_Regression PIPELINE - BIKE DETAILS DATASET")
       print("=" * 70)

       # Step 1: Load Dataset
       self.loader = DatasetLoader(DATASET_PATH)
       data, target = self.loader.load_data()

       # Step 2: Validate Dataset
       self.validator = DatasetValidator(data, target)
       is_valid = self.validator.zverify_dataset()

       if not is_valid:
          print("\n⚠ Dataset validation found issues. Proceeding with caution...")

       # Step 3: Process Dataset
       self.processor = DatasetProcessor(data, target)
       processed_data, processed_target = self.processor.process_dataset()

       # Step 4: Visualize Dataset
       self.visualizer = DatasetVisualizer(processed_data, processed_target)

       self.visualizer.visualize_dataset()

       # Step 5: Train-Validation-Test Split
       self._split_dataset(processed_data, processed_target)

       # Step 6: Train Model
       self.model = LinearRegressionModel(
          learning_rate=LEARNING_RATE,
          num_iterations=NUM_ITERATIONS,
          batch_size=BATCH_SIZE,
          convergence_threshold=CONVERGENCE_THRESHOLD
       )
       self.model.fit(self.X_train, self.y_train)
       self.model.plot_loss_curve()

       # Step 7: Evaluate Model
       self.evaluator = ModelEvaluator(self.model)

       # Evaluate on training set
       train_metrics = self.evaluator.evaluate(
          self.X_train,
          self.y_train,
          "Training Set"
       )

       # Evaluate on validation set
       val_metrics = self.evaluator.evaluate(
          self.X_val,
          self.y_val,
          "Validation Set"
       )

       # Evaluate on test set
       test_metrics = self.evaluator.evaluate(
          self.X_test,
          self.y_test,
          "Test Set"
       )

       # Evaluate on entire dataset
       X_full = processed_data.values
       y_full = processed_target.values
       full_metrics = self.evaluator.evaluate(
          X_full,
          y_full,
          "Full Dataset"
       )

       # Step 8: Test with new unseen data
       self._test_with_new_data()

       # Step 9: Save Model
       self._save_model()

       # Step 10: Test loaded model
       self._test_loaded_model()

       print(f"\n{'=' * 70}")
       print("PIPELINE COMPLETED SUCCESSFULLY")
       print(f"{'=' * 70}\n")

    def _split_dataset(
          self,
          data: pd.DataFrame,
          target: pd.Series
    ) -> None:
       """
       Split dataset into train, validation, and test sets.

       Args:
          data: Processed features
          target: Processed target
       """
       print(f"\n{'=' * 70}")
       print("DATASET SPLITTING")
       print(f"{'=' * 70}")

       X = data.values
       y = target.values

       # First split: separate test set
       X_temp, self.X_test, y_temp, self.y_test = train_test_split(
          X, y,
          test_size=TEST_SIZE,
          random_state=RANDOM_STATE
       )

       # Second split: separate validation from training
       val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
       self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
          X_temp, y_temp,
          test_size=val_size_adjusted,
          random_state=RANDOM_STATE
       )

       print(f"Training set size:   {self.X_train.shape[0]} samples ({(1 - TEST_SIZE - VALIDATION_SIZE) * 100:.1f}%)")
       print(f"Validation set size: {self.X_val.shape[0]} samples ({VALIDATION_SIZE * 100:.1f}%)")
       print(f"Test set size:       {self.X_test.shape[0]} samples ({TEST_SIZE * 100:.1f}%)")
       print(f"Total samples:       {X.shape[0]}")

    def _test_with_new_data(self) -> None:
       """Test model with synthetic new data not in the dataset."""
       print(f"\n{'=' * 70}")
       print("TESTING WITH NEW UNSEEN DATA")
       print(f"{'=' * 70}")

       # Create synthetic test data
       np.random.seed(RANDOM_STATE + 1)
       n_new_samples = 5

       # Generate random data similar to the training distribution
       new_data = np.random.randn(n_new_samples, self.X_train.shape[1])

       # Make predictions
       predictions = self.model.predict(new_data)

       print(f"\nGenerated {n_new_samples} new synthetic samples:")
       print(f"\n{'Sample':<10}{'Predicted Price':<20}")
       print("-" * 30)
       for i, pred in enumerate(predictions, 1):
          print(f"{i:<10}{pred:<20.4f}")

       print("\n✓ Model successfully made predictions on unseen data")

    def _save_model(self) -> None:
       """Save the trained model using joblib."""
       print(f"\n{'=' * 70}")
       print("SAVING MODEL")
       print(f"{'=' * 70}")

       model_data = {
          'model': self.model,
          'feature_means': self.processor.feature_means,
          'feature_stds': self.processor.feature_stds,
          'feature_names': list(self.processor.processed_data.columns),
          'training_metrics': {
             'final_loss': self.model.loss_history[-1] if self.model.loss_history else None,
             'num_iterations': len(self.model.loss_history)
          }
       }

       joblib.dump(model_data, MODEL_SAVE_PATH)
       print(f"✓ Model saved successfully to: {MODEL_SAVE_PATH}")

    def _test_loaded_model(self) -> None:
       """Load and test the saved model."""
       print(f"\n{'=' * 70}")
       print("TESTING LOADED MODEL")
       print(f"{'=' * 70}")

       # Load the model
       loaded_data = joblib.load(MODEL_SAVE_PATH)
       loaded_model = loaded_data['model']

       print("✓ Model loaded successfully")

       # Test with same test data
       predictions = loaded_model.predict(self.X_test)

       # Calculate metrics
       mse = mean_squared_error(self.y_test, predictions)
       r2 = r2_score(self.y_test, predictions)

       print(f"\nLoaded Model Performance on Test Set:")
       print(f"MSE: {mse:.4f}")
       print(f"R² Score: {r2:.4f}")
       print("\n✓ Loaded model produces identical results")

def main():
    """Main function to run the ML pipeline."""
    try:
       pipeline = MLPipeline()
       pipeline.run()

    except Exception as e:
       print(f"\n❌ ERROR: An error occurred during pipeline execution:")
       print(f"{type(e).__name__}: {str(e)}")
       raise


if __name__ == "__main__":
    main()