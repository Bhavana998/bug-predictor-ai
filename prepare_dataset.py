"""
Prepare a balanced dataset with both bugs and features
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

print("="*70)
print("📊 PREPARING BALANCED DATASET")
print("="*70)

# Load original data
original_df = pd.read_csv('data/bug_data.csv')
print(f"\n📂 Loaded {len(original_df)} bug reports")

# Feature templates
feature_templates = [
    "Add new {} functionality",
    "Implement {} feature",
    "Create {} dashboard",
    "Develop {} integration",
    "Enhance {} UI/UX",
    "Add support for {}",
    "Improve {} performance",
    "Implement {} caching",
    "Add {} validation",
    "Create {} API endpoint",
    "Develop {} reporting module",
    "Add {} export feature",
    "Implement {} search",
    "Create {} notification system",
    "Add {} authentication method",
    "Improve {} error handling",
    "Implement {} backup system",
    "Add {} configuration options",
    "Create {} documentation",
    "Develop {} mobile support"
]

feature_contexts = [
    "user authentication", "data visualization", "search", "dashboard", "reporting",
    "email notifications", "file upload", "REST API", "database", "user profile",
    "admin panel", "help system", "analytics", "payment", "shopping cart",
    "inventory", "customer support", "billing", "subscription", "multi-language",
    "dark mode", "accessibility", "performance", "backup", "export"
]

print("\n🔄 Creating synthetic feature requests...")

# Create feature dataframe
feature_data = []

for i in range(len(original_df)):  # Create same number as bugs
    # Generate feature summary
    template = np.random.choice(feature_templates)
    context = np.random.choice(feature_contexts)
    summary = template.format(context)
    
    # Create description
    description = f"""
    Feature Request: {summary}
    
    Description:
    This feature would allow users to {context} more efficiently.
    
    Benefits:
    - Improved user experience
    - Better productivity
    - Enhanced functionality
    
    Acceptance Criteria:
    1. Feature works as specified
    2. Performance is acceptable
    3. Documentation is updated
    4. Tests are passing
    """
    
    feature_data.append({
        'Summary': summary,
        'Description': description.strip(),
        'Status': 'RESOLVED',
        'Resolution': 'FIXED',
        'Severity': 'enhancement',
        'Priority': 'P3'
    })

# Create feature dataframe
feature_df = pd.DataFrame(feature_data)

print(f"✅ Created {len(feature_df)} feature requests")

# Add remaining columns from original (with defaults)
for col in original_df.columns:
    if col not in feature_df.columns:
        feature_df[col] = original_df[col].iloc[0] if len(original_df) > 0 else None

# Combine datasets
balanced_df = pd.concat([original_df, feature_df], ignore_index=True)

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = 'data/balanced_bug_data.csv'
balanced_df.to_csv(output_path, index=False)
print(f"\n✅ Saved balanced dataset to {output_path}")
print(f"   Total records: {len(balanced_df)}")
print(f"   Bugs: {len(original_df)} (50%)")
print(f"   Features: {len(feature_df)} (50%)")