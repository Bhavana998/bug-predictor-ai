"""
Train model on balanced bug/feature dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 TRAINING BUG PREDICTION MODEL ON BALANCED DATASET")
print("="*80)

# Create models directory
os.makedirs('models', exist_ok=True)

# Load balanced dataset
dataset_path = 'data/balanced_bug_data.csv'
if not os.path.exists(dataset_path):
    print(f"\n❌ Dataset not found at {dataset_path}")
    print("Running dataset preparation first...")
    exec(open('prepare_dataset.py').read())

df = pd.read_csv(dataset_path)
print(f"\n✅ Loaded {len(df)} records from balanced dataset")

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple tokenization (keep words with length > 2)
    words = [word for word in text.split() if len(word) > 2]
    
    return ' '.join(words)

# Create labels based on multiple criteria
def create_labels(df):
    """Create binary labels: Bug vs Feature"""
    print("\n🏷️ Creating labels...")
    
    labels = []
    
    # Bug indicators
    bug_statuses = ['NEW', 'CONFIRMED', 'ASSIGNED', 'REOPENED']
    bug_resolutions = ['FIXED', 'WORKSFORME', 'DUPLICATE', 'INVALID']
    bug_severities = ['blocker', 'critical', 'major', 'normal']
    
    # Feature indicators
    feature_statuses = ['RESOLVED', 'CLOSED', 'VERIFIED']
    feature_resolutions = ['FIXED']  # Fixed features
    feature_severities = ['enhancement', 'feature', 'trivial']
    
    bug_count = 0
    feature_count = 0
    
    for idx, row in df.iterrows():
        is_bug = True  # Default to bug
        
        # Check Status
        if pd.notna(row.get('Status')):
            status = str(row['Status']).upper()
            if status in feature_statuses:
                is_bug = False
        
        # Check Severity
        if pd.notna(row.get('Severity')):
            severity = str(row['Severity']).lower()
            if severity in feature_severities:
                is_bug = False
        
        # Check Summary for feature keywords
        if pd.notna(row.get('Summary')):
            summary = str(row['Summary']).lower()
            feature_keywords = ['add', 'implement', 'improve', 'create', 'enhance', 
                               'develop', 'support', 'feature', 'new']
            if any(keyword in summary for keyword in feature_keywords):
                # But check if it's actually a bug report
                bug_keywords = ['crash', 'error', 'exception', 'fail', 'bug', 'issue']
                if not any(bug in summary for bug in bug_keywords):
                    is_bug = False
        
        if is_bug:
            labels.append('Bug')
            bug_count += 1
        else:
            labels.append('Feature/Enhancement')
            feature_count += 1
    
    df['label'] = labels
    print(f"   Bugs: {bug_count} ({bug_count/len(df)*100:.1f}%)")
    print(f"   Features: {feature_count} ({feature_count/len(df)*100:.1f}%)")
    
    return df

# Create labels
df = create_labels(df)

# Check if we have both classes
if len(df['label'].unique()) < 2:
    print("\n❌ ERROR: Still only one class!")
    print(f"Labels: {df['label'].unique()}")
    exit(1)

# Use Summary as primary text column
text_column = 'Summary'
print(f"\n📝 Using text column: {text_column}")

# Preprocess texts
print("\n🔧 Preprocessing text data...")
processed_texts = []

for i, text in enumerate(df[text_column]):
    if i % 50 == 0 and i > 0:
        print(f"   Processed {i}/{len(df)} texts")
    processed = preprocess_text(text)
    processed_texts.append(processed)

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
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   Feature matrix shape: {X_train_tfidf.shape}")
print(f"   Number of features: {len(vectorizer.get_feature_names_out())}")

# Initialize models
print("\n🤖 Initializing models...")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
print("   ✅ Random Forest initialized")

# SVM
svm_model = SVC(
    C=10.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42,
    class_weight='balanced'
)
print("   ✅ SVM initialized")

# Logistic Regression
lr_model = LogisticRegression(
    C=2.0,
    max_iter=2000,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
print("   ✅ Logistic Regression initialized")

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=15,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
print("   ✅ XGBoost initialized")

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=15,
    learning_rate=0.05,
    num_leaves=50,
    random_state=42,
    verbose=-1,
    class_weight='balanced'
)
print("   ✅ LightGBM initialized")

# Create ensemble
print("\n🔄 Creating ensemble model...")
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('svm', svm_model),
        ('lr', lr_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    voting='soft',
    weights=[3, 2, 1, 4, 3]  # Give more weight to better models
)

print(f"✅ Ensemble created with 5 base models")

# Train ensemble
print("\n🚀 Training ensemble model...")
ensemble_model.fit(X_train_tfidf, y_train)

# Make predictions
print("\n📊 Evaluating model...")
y_pred = ensemble_model.predict(X_test_tfidf)
y_proba = ensemble_model.predict_proba(X_test_tfidf)

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

# Classification Report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save models
print("\n💾 Saving models...")
joblib.dump(ensemble_model, 'models/ensemble_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("✅ Models saved to 'models/' directory")

# Test with examples
print("\n🔮 Testing with examples:")

test_examples = [
    "Application crashes when user clicks submit button",
    "Add dark mode support for better user experience",
    "Null pointer exception in authentication module",
    "Implement export to PDF functionality",
    "Database connection timeout after 30 seconds",
    "Create new dashboard for analytics",
    "Memory leak in background service",
    "Improve search performance with indexing"
]

for text in test_examples:
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    pred = ensemble_model.predict(X)[0]
    proba = ensemble_model.predict_proba(X)[0]
    pred_label = label_encoder.inverse_transform([pred])[0]
    confidence = max(proba)
    
    # Determine actual class for display
    actual = "Bug" if any(word in text.lower() for word in ['crash', 'error', 'exception', 'timeout', 'leak']) else "Feature"
    
    print(f"\n   Text: {text}")
    print(f"   Actual: {actual}")
    print(f"   Predicted: {pred_label} (Confidence: {confidence:.2%})")
    if pred_label == actual:
        print(f"   ✅ Correct!")
    else:
        print(f"   ❌ Incorrect")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)
print("\nRun: streamlit run app.py")