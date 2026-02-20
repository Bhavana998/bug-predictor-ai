"""
Severity Prediction Module - Complete Working Version
Predicts: Low, Medium, High, Critical severity levels
Auto-trains on first use
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import re

class SeverityPredictor:
    """Predict severity of bugs/features with auto-training"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.severity_labels = ['Low', 'Medium', 'High', 'Critical']
        self.is_fitted = False
        self.training_data = self.create_training_data()
        
    def create_training_data(self):
        """Create comprehensive training data"""
        
        # Critical severity examples
        critical_texts = [
            "CRITICAL: NullPointerException causing application crash for all users",
            "Security vulnerability allowing SQL injection in login form",
            "Database deadlock occurring every minute in production",
            "Memory leak causing OutOfMemoryError every 2 hours",
            "Production outage - API returning 500 errors for 50% of requests",
            "Data corruption in customer records due to race condition",
            "Authentication bypass vulnerability exposing user data",
            "System crash on startup - cannot initialize database connection",
            "Heap memory leak - 2GB increase over 4 hours, service crashes",
            "Deadlock in transaction processing - system unresponsive"
        ]
        
        # High severity examples
        high_texts = [
            "NullPointerException in login module when password is empty",
            "API timeout after 30 seconds - users experiencing delays",
            "Data not saving correctly to database",
            "Payment processing failing for credit cards",
            "Search functionality returning incorrect results",
            "File upload fails for files > 10MB",
            "User session expiring before timeout period",
            "Email notifications not being sent",
            "Database connection pool exhausted",
            "Authentication failing for valid users"
        ]
        
        # Medium severity examples
        medium_texts = [
            "UI layout broken on mobile devices",
            "Button not responding on first click",
            "Text overlapping in settings panel",
            "Dropdown menu flickering on hover",
            "Progress bar not updating correctly",
            "Modal dialog closing unexpectedly",
            "Form validation showing wrong error messages",
            "Color contrast issues in dark mode",
            "Font size inconsistent across pages",
            "Animation stuttering on scroll"
        ]
        
        # Low severity examples
        low_texts = [
            "Add dark mode support",
            "Improve search performance",
            "Implement export to PDF feature",
            "Add user profile customization",
            "Create analytics dashboard",
            "Update documentation",
            "Add tooltips for better UX",
            "Implement keyboard shortcuts",
            "Enhance error messages",
            "Add loading indicators"
        ]
        
        # Create features and labels
        X_texts = []
        y_labels = []
        
        # Add critical (3)
        for text in critical_texts:
            X_texts.append(text)
            y_labels.append(3)
            # Add variations
            for _ in range(3):
                X_texts.append(text + f" variant {np.random.randint(100)}")
                y_labels.append(3)
        
        # Add high (2)
        for text in high_texts:
            X_texts.append(text)
            y_labels.append(2)
            for _ in range(3):
                X_texts.append(text + f" variant {np.random.randint(100)}")
                y_labels.append(2)
        
        # Add medium (1)
        for text in medium_texts:
            X_texts.append(text)
            y_labels.append(1)
            for _ in range(2):
                X_texts.append(text + f" variant {np.random.randint(100)}")
                y_labels.append(1)
        
        # Add low (0)
        for text in low_texts:
            X_texts.append(text)
            y_labels.append(0)
            for _ in range(2):
                X_texts.append(text + f" variant {np.random.randint(100)}")
                y_labels.append(0)
        
        return X_texts, y_labels
    
    def extract_features(self, text):
        """Extract features from text for severity prediction"""
        if not isinstance(text, str):
            text = str(text) if text else ""
        
        text_lower = text.lower()
        features = {}
        
        # Critical keywords (weight: 4)
        critical_keywords = [
            'crash', 'deadlock', 'security', 'vulnerability', 'outage',
            'data loss', 'corruption', 'exploit', 'breach', 'downtime',
            'production', 'emergency', 'critical', 'catastrophic',
            'all users', '100%', 'complete failure', 'system down'
        ]
        
        # High keywords (weight: 3)
        high_keywords = [
            'error', 'exception', 'timeout', 'failed', 'broken',
            'incorrect', 'wrong', 'issue', 'bug', 'problem',
            'major', 'severe', 'significant', 'data corruption'
        ]
        
        # Medium keywords (weight: 2)
        medium_keywords = [
            'minor', 'cosmetic', 'ui', 'layout', 'display',
            'typo', 'spelling', 'format', 'style', 'appearance',
            'visual', 'alignment', 'spacing', 'color'
        ]
        
        # Low keywords (weight: 1)
        low_keywords = [
            'suggestion', 'enhancement', 'improvement', 'feature',
            'request', 'would be nice', 'consider', 'maybe',
            'could be better', 'nice to have', 'option'
        ]
        
        # Count keywords with weights
        features['critical_score'] = sum(4 for k in critical_keywords if k in text_lower)
        features['high_score'] = sum(3 for k in high_keywords if k in text_lower)
        features['medium_score'] = sum(2 for k in medium_keywords if k in text_lower)
        features['low_score'] = sum(1 for k in low_keywords if k in text_lower)
        
        # Text characteristics
        features['length'] = min(len(text), 1000) / 100
        features['word_count'] = min(len(text.split()), 200) / 20
        features['has_stack_trace'] = 1 if ('stack trace' in text_lower or 
                                            'at ' in text_lower and '.' in text_lower) else 0
        features['has_exception'] = 1 if ('exception' in text_lower or 
                                          'error:' in text_lower) else 0
        features['has_crash'] = 1 if 'crash' in text_lower else 0
        features['has_deadlock'] = 1 if 'deadlock' in text_lower else 0
        features['has_security'] = 1 if any(word in text_lower for word in 
                                           ['security', 'vulnerability', 'exploit', 'breach']) else 0
        features['has_suggestion'] = 1 if any(word in text_lower for word in 
                                             ['suggest', 'feature', 'enhance', 'improve']) else 0
        
        # Punctuation features
        features['exclamation_count'] = min(text.count('!'), 10)
        features['question_count'] = min(text.count('?'), 10)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        
        return features
    
    def train(self):
        """Train the severity model"""
        print("Training severity predictor...")
        
        X_texts, y_labels = self.training_data
        
        # Extract features
        X_features = []
        for text in X_texts:
            X_features.append(self.extract_features(text))
        
        X = pd.DataFrame(X_features)
        X = X.fillna(0)
        
        # Train model
        self.model.fit(X, y_labels)
        self.is_fitted = True
        print("✅ Severity predictor trained successfully")
        
        # Save model
        self.save()
        
    def predict(self, texts):
        """Predict severity for texts"""
        if not self.is_fitted:
            self.train()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Extract features
        X_features = []
        for text in texts:
            X_features.append(self.extract_features(text))
        
        X = pd.DataFrame(X_features)
        X = X.fillna(0)
        
        # Predict
        predictions = self.model.predict(X)
        
        # Convert to labels
        return [self.severity_labels[int(p)] for p in predictions]
    
    def predict_proba(self, texts):
        """Get probability distribution for each severity level"""
        if not self.is_fitted:
            self.train()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Extract features
        X_features = []
        for text in texts:
            X_features.append(self.extract_features(text))
        
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
    
    def save(self, path='models/severity_model.pkl'):
        """Save trained model"""
        if self.is_fitted:
            os.makedirs('models', exist_ok=True)
            joblib.dump({
                'model': self.model,
                'is_fitted': self.is_fitted,
                'severity_labels': self.severity_labels
            }, path)
            print(f"✅ Severity model saved to {path}")
    
    def load(self, path='models/severity_model.pkl'):
        """Load trained model"""
        try:
            if os.path.exists(path):
                data = joblib.load(path)
                self.model = data['model']
                self.is_fitted = data['is_fitted']
                self.severity_labels = data['severity_labels']
                print(f"✅ Severity model loaded from {path}")
                return True
        except:
            pass
        return False