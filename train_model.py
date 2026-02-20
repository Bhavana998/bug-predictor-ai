"""
Train model on balanced dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 TRAINING BUG PREDICTION MODEL")
print("="*80)

# Create models directory
os.makedirs('models', exist_ok=True)

# Check if balanced dataset exists
if not os.path.exists('data/balanced_bug_data.csv'):
    print("\n⚠️ Balanced dataset not found. Running prepare_dataset.py first...")
    os.system('python prepare_dataset.py')

# Load balanced dataset
df = pd.read_csv('data/balanced_bug_data.csv')
print(f"\n✅ Loaded {len(df)} records from balanced dataset")

# Text preprocessing function
def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create labels
print("\n🏷️ Creating labels...")

labels = []
bug_count = 0
feature_count = 0

for idx, row in df.iterrows():
    summary = str(row.get('Summary', '')).lower()
    status = str(row.get('Status', '')).upper()
    
    # Feature indicators
    feature_keywords = ['add', 'implement', 'improve', 'create', 'enhance', 
                       'develop', 'support', 'feature', 'new']
    
    # Bug indicators
    bug_statuses = ['NEW', 'CONFIRMED', 'ASSIGNED', 'REOPENED']
    
    # Determine label
    if any(keyword in summary for keyword in feature_keywords) and status not in bug_statuses:
        labels.append('Feature/Enhancement')
        feature_count += 1
    else:
        labels.append('Bug')
        bug_count += 1

df['label'] = labels

print(f"   Bugs: {bug_count} ({bug_count/len(df)*100:.1f}%)")
print(f"   Features: {feature_count} ({feature_count/len(df)*100:.1f}%)")

# Preprocess texts
print("\n🔧 Preprocessing text...")
processed_texts = []

for i, text in enumerate(df['Summary']):
    if i % 50 == 0 and i > 0:
        print(f"   Processed {i}/{len(df)} texts")
    cleaned = clean_text(text)
    processed_texts.append(cleaned)

df['processed_text'] = processed_texts

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

print(f"\n🏷️ Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"   {i}: {label}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['label_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_encoded']
)

print(f"\n📊 Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Create TF-IDF features
print("\n📊 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"   Feature matrix shape: {X_train_vec.shape}")
print(f"   Number of features: {len(vectorizer.get_feature_names_out())}")

# Create models
print("\n🤖 Creating models...")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
print("   ✅ Random Forest")

# SVM
svm_model = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)
print("   ✅ SVM")

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
print("   ✅ Logistic Regression")

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
print("   ✅ XGBoost")

# Create ensemble
print("\n🔄 Creating ensemble...")
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('lr', lr_model),
        ('xgb', xgb_model)
    ],
    voting='soft'
)

# Train
print("\n🚀 Training ensemble...")
ensemble_model.fit(X_train_vec, y_train)

# Evaluate
print("\n📊 Evaluating model...")
y_pred = ensemble_model.predict(X_test_vec)
y_proba = ensemble_model.predict_proba(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"      {cm[0]}")
print(f"      {cm[1]}")

# Save models
print("\n💾 Saving models...")
joblib.dump(ensemble_model, 'models/ensemble_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

# Save metrics
import json
metrics = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1)
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Models saved to 'models/' directory")

# Test with examples
print("\n🔮 Testing with examples:")
test_examples = [
    "Application crashes when user clicks submit",
    "Add dark mode support for better UX",
    "Null pointer exception in login module",
    "Implement export to PDF feature"
]

for text in test_examples:
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = ensemble_model.predict(X)[0]
    proba = ensemble_model.predict_proba(X)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    confidence = max(proba)
    
    print(f"\n   Text: {text}")
    print(f"   Predicted: {pred_label} (Confidence: {confidence:.2%})")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nNow run: streamlit run app.py")