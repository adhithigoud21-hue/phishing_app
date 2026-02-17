# Phishing URL Detection Configuration

# Model Training Settings
MIN_TRAINING_SAMPLES = 30000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Hyperparameters
# Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5

# Gradient Boosting
GB_N_ESTIMATORS = 150
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 7

# XGBoost
XGB_N_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 7

# Feature Extraction Settings
FEATURE_COUNT = 56
EXTRACT_DNS_FEATURES = False  # Set to True if you want DNS lookups (slower)

# Streamlit App Settings
APP_TITLE = "Advanced Phishing URL Detector"
APP_ICON = "🔒"
THEME = "wide"

# Paths
DATA_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = "phishing_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"
METRICS_PATH = "model_metrics.pkl"
FEATURE_IMPORTANCE_PATH = "feature_importance.csv"

# Thresholds
PHISHING_THRESHOLD = 0.5  # Probability threshold for phishing classification
HIGH_RISK_THRESHOLD = 0.7
LOW_RISK_THRESHOLD = 0.3
