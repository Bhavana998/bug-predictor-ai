"""
Severity Prediction Module - Predicts Low, Medium, High, Critical severity
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import re

class SeverityPredictor:
    """Predict severity of bugs/features"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.severity_labels = ['Low', 'Medium', 'High', 'Critical']
        self.is_fitted = False
        self._train_sample_data()
    
    def _train_sample_data(self):
        """Train with sample data"""
        # Sample training data
        X_texts = [
            # Critical examples
            "Critical NullPointerException causing application crash for all users",
            "Security vulnerability allowing SQL injection in login form",
            "Database deadlock occurring every minute in production",
            "Memory leak causing OutOfMemoryError every 2 hours",
            # High examples
            "NullPointerException in login module when password is empty",
            "API timeout after 30 seconds - users experiencing delays",
            "Data not saving correctly to database",
            # Medium examples
            "UI layout broken on mobile devices",
            "Button not responding on first click",
            "Text overlapping in settings panel",
            # Low examples
            "Add dark mode support",
            "Improve search performance",
            "Implement export to PDF feature"
        ]
        
        y_labels = [3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0]  # 3=Critical, 2=High, 1=Medium, 0=Low
        
        # Extract features
        X_features = [self._extract_features(text) for text in X_texts]
        X = pd.DataFrame(X_features)
        X = X.fillna(0)
        
        # Train
        self.model.fit(X, y_labels)
        self.is_fitted = True
    
    def _extract_features(self, text):
        """Extract features from text"""
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        text_lower = text.lower()
        features = {}
        
        # Critical keywords
        critical_keywords = ['crash', 'deadlock', 'security', 'vulnerability', 'outage',
                            'data loss', 'corruption', 'exploit', 'breach', 'critical']
        
        # High keywords
        high_keywords = ['error', 'exception', 'timeout', 'failed', 'broken',
                        'incorrect', 'wrong', 'major', 'severe']
        
        # Medium keywords
        medium_keywords = ['bug', 'issue', 'problem', 'minor', 'cosmetic', 'ui',
                          'layout', 'display', 'typo']
        
        # Low keywords
        low_keywords = ['suggestion', 'enhancement', 'improvement', 'feature',
                       'request', 'nice', 'consider', 'maybe']
        
        # Count keywords
        features['critical_score'] = sum(1 for k in critical_keywords if k in text_lower)
        features['high_score'] = sum(1 for k in high_keywords if k in text_lower)
        features['medium_score'] = sum(1 for k in medium_keywords if k in text_lower)
        features['low_score'] = sum(1 for k in low_keywords if k in text_lower)
        
        # Other features
        features['length'] = min(len(text), 500) / 100
        features['has_exception'] = 1 if 'exception' in text_lower else 0
        features['has_error'] = 1 if 'error' in text_lower else 0
        features['has_crash'] = 1 if 'crash' in text_lower else 0
        
        return features
    
    def predict(self, texts):
        """Predict severity for texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Extract features
        X_features = [self._extract_features(text) for text in texts]
        X = pd.DataFrame(X_features)
        X = X.fillna(0)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Convert to labels
        return [self.severity_labels[int(p)] for p in predictions]
    
    def predict_proba(self, texts):
        """Get probability distribution for each severity level"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Extract features
        X_features = [self._extract_features(text) for text in texts]
        X = pd.DataFrame(X_features)
        X = X.fillna(0)
        
        # Get probabilities
        probas = self.model.predict_proba(X)
        
        # Format results
        results = []
        for i in range(len(texts)):
            probs = {}
            for j, label in enumerate(self.severity_labels):
                probs[label] = probas[i][j]
            results.append(probs)
        
        return results