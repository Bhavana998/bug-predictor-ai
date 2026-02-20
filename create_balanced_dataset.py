"""
Create a balanced dataset with both bugs and features
"""

import pandas as pd
import numpy as np
import os
import shutil

print("="*60)
print("🔄 CREATING BALANCED DATASET")
print("="*60)

# Load original data
original_df = pd.read_csv('data/bug_data.csv')
print(f"\n📊 Original dataset: {len(original_df)} rows (all bugs)")

# Create synthetic feature requests
feature_templates = [
    "Add new feature for {}",
    "Implement {} functionality",
    "Improve {} performance",
    "Create {} dashboard",
    "Add support for {}",
    "Enhance {} UI/UX",
    "Develop {} module",
    "Integrate {} service",
    "Update {} documentation",
    "Refactor {} code",
    "Optimize {} queries",
    "Add {} API endpoint",
    "Create {} test suite",
    "Implement {} caching",
    "Add {} validation",
    "Improve {} security",
    "Update {} dependencies",
    "Add {} monitoring",
    "Create {} backup system",
    "Implement {} logging"
]

feature_contexts = [
    "user authentication", "data export", "search", "dashboard", "reporting",
    "email notifications", "file upload", "API", "database", "cache",
    "frontend", "backend", "mobile app", "web interface", "admin panel",
    "user profile", "settings", "help system", "analytics", "billing",
    "subscription", "payment", "shipping", "inventory", "customer support"
]

# Create feature requests
feature_requests = []
for i in range(len(original_df)):  # Create same number of features as bugs
    template = np.random.choice(feature_templates)
    context = np.random.choice(feature_contexts)
    feature = template.format(context)
    feature_requests.append(feature)

# Create feature dataframe (copy structure but modify content)
feature_df = original_df.copy()

# Modify relevant columns for features
feature_df['Summary'] = feature_requests
feature_df['Description'] = [f"Implementation request: {f}" for f in feature_requests]

# Set status to appropriate for features
if 'Status' in feature_df.columns:
    feature_df['Status'] = 'RESOLVED'

if 'Resolution' in feature_df.columns:
    feature_df['Resolution'] = 'FIXED'

if 'Severity' in feature_df.columns:
    feature_df['Severity'] = 'enhancement'

# Combine datasets
balanced_df = pd.concat([original_df, feature_df], ignore_index=True)

print(f"\n📊 Balanced dataset: {len(balanced_df)} rows")
print(f"   Original bugs: {len(original_df)}")
print(f"   Synthetic features: {len(feature_df)}")

# Save balanced dataset
balanced_df.to_csv('data/balanced_bug_data.csv', index=False)
print(f"\n✅ Saved balanced dataset to data/balanced_bug_data.csv")

# Show sample
print(f"\n📝 Sample bugs (original):")
for i in range(3):
    print(f"   {i+1}. {original_df['Summary'].iloc[i][:100]}...")

print(f"\n📝 Sample features (synthetic):")
for i in range(3):
    print(f"   {i+1}. {feature_df['Summary'].iloc[i][:100]}...")