"""
Configuration settings for the Bug Prediction Model
"""

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1
}

# Text Preprocessing Configuration
TEXT_CONFIG = {
    'max_features': 5000,
    'ngram_range': (1, 3),
    'min_df': 2,
    'max_df': 0.95,
    'use_idf': True,
    'sublinear_tf': True
}

# Ensemble Model Parameters
ENSEMBLE_PARAMS = {
    'voting': 'soft',
    'weights': [2, 2, 1, 3]  # Weights for [Random Forest, SVM, Logistic Regression, XGBoost]
}

# Base Model Parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'random_state': 42
}

LR_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# File Paths
MODEL_PATH = 'models/ensemble_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
DATA_PATH = 'data/bug_data.csv'