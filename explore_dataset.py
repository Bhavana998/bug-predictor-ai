"""
explore_data.py - Script to explore your bug dataset
Run this first to understand your data structure
"""

import pandas as pd
import os

def explore_dataset():
    print("\n" + "="*70)
    print("🔍 BUG DATASET EXPLORATION TOOL")
    print("="*70)
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("\n❌ 'data' folder not found!")
        print("Creating data folder...")
        os.makedirs('data', exist_ok=True)
    
    # Look for CSV files
    print("\n📁 Looking for data files...")
    csv_files = []
    if os.path.exists('data'):
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    if not csv_files:
        print("\n❌ No CSV files found in 'data' folder!")
        print("\nPlease place your dataset file in the 'data' folder.")
        print("Expected path: data/bug_data.csv")
        return
    
    print(f"\n✅ Found CSV files: {csv_files}")
    
    # Let user choose which file to explore
    if len(csv_files) > 1:
        print("\nMultiple files found. Please choose:")
        for i, file in enumerate(csv_files, 1):
            print(f"   {i}. {file}")
        choice = input("\nEnter number (or press Enter for first file): ").strip()
        try:
            idx = int(choice) - 1 if choice else 0
            filename = csv_files[idx]
        except:
            filename = csv_files[0]
    else:
        filename = csv_files[0]
    
    filepath = os.path.join('data', filename)
    print(f"\n📂 Exploring: {filename}")
    
    try:
        # Load the data
        print("\n⏳ Loading data...")
        df = pd.read_csv(filepath)
        
        # Basic info
        print("\n" + "="*70)
        print("📊 BASIC INFORMATION")
        print("="*70)
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Column names
        print("\n" + "="*70)
        print("📋 COLUMN NAMES")
        print("="*70)
        for i, col in enumerate(df.columns, 1):
            print(f"{i:3d}. {col}")
        
        # Data types
        print("\n" + "="*70)
        print("📈 DATA TYPES")
        print("="*70)
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        # Check for text columns
        print("\n" + "="*70)
        print("📝 TEXT COLUMNS (for features)")
        print("="*70)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if it contains text (long strings)
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
                if isinstance(sample, str) and len(sample) > 20:
                    text_columns.append(col)
                    print(f"\n📌 {col}:")
                    print(f"   Non-null: {df[col].notna().sum()}/{len(df)}")
                    print(f"   Sample: {sample[:200]}...")
        
        # Check for label columns
        print("\n" + "="*70)
        print("🏷️ POTENTIAL LABEL COLUMNS")
        print("="*70)
        
        label_keywords = ['status', 'resolution', 'severity', 'priority', 'type', 'classification', 'bug', 'issue']
        label_columns = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in label_keywords):
                label_columns.append(col)
                print(f"\n📌 {col}:")
                print(f"   Unique values: {df[col].nunique()}")
                print(f"   Top values:")
                for val, count in df[col].value_counts().head(5).items():
                    print(f"      {val}: {count} ({count/len(df)*100:.1f}%)")
        
        # Summary and recommendations
        print("\n" + "="*70)
        print("✅ RECOMMENDATIONS")
        print("="*70)
        
        print("\nBased on your data, here's what to use:")
        
        if text_columns:
            print(f"\n📝 For TEXT FEATURES (use one of these):")
            for i, col in enumerate(text_columns[:3], 1):
                print(f"   {i}. {col}")
            print(f"\nRecommended: {text_columns[0]}")
        else:
            print("\n❌ No good text columns found!")
        
        if label_columns:
            print(f"\n🏷️ For LABEL (target variable):")
            for i, col in enumerate(label_columns[:3], 1):
                print(f"   {i}. {col}")
            
            # Check which column might be best
            best_label = None
            best_score = 0
            for col in label_columns:
                unique_ratio = df[col].nunique() / len(df)
                if 0.1 < unique_ratio < 0.5:  # Good balance
                    score = 2
                else:
                    score = 1
                
                # Check if it has bug-related terms
                if any(term in str(df[col].iloc[0]).lower() for term in ['bug', 'error', 'fix']):
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_label = col
            
            print(f"\nRecommended: {best_label}")
        else:
            print("\n❌ No good label columns found!")
        
        # Save exploration results
        print("\n" + "="*70)
        print("💾 SAVING RESULTS")
        print("="*70)
        
        # Create a summary file
        summary = {
            'filename': filename,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'text_columns': text_columns[:3],
            'label_columns': label_columns[:3],
            'recommended_text': text_columns[0] if text_columns else None,
            'recommended_label': best_label if label_columns else None
        }
        
        import json
        with open('data_exploration_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("✅ Exploration summary saved to 'data_exploration_summary.json'")
        
        print("\n" + "="*70)
        print("🎉 EXPLORATION COMPLETE!")
        print("="*70)
        print("\nYou can now update train_model.py with these column names.")
        print("Or run: python train_model.py")
        
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        print("\nPlease check that your CSV file is properly formatted.")

if __name__ == "__main__":
    explore_dataset()