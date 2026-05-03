import pandas as pd
from collections import Counter
import os
from pathlib import Path

def audit_csv_files(test_path="labels_test.csv", train_path="labels_train.csv"):
    print("=" * 60)
    print("BEE DATASET AUDIT SCRIPT")
    print("=" * 60)
    
    # ====================== TEST FILE ======================
    print("\nTEST FILE ANALYSIS")
    print("-" * 40)
    
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
    else:
        test_df = pd.read_csv(test_path)
        print(f"Loaded test file: {test_df.shape[0]:,} annotations, {test_df['image name'].nunique():,} unique images")
        
        print(f"\nTest Columns: {test_df.columns.tolist()}")
        
        # Class analysis
        print("\nClass Distribution (Test):")
        class_count_test = test_df['class name'].value_counts()
        for cls, count in class_count_test.items():
            print(f"  {cls:12} : {count:6,}  ({count/len(test_df)*100:5.2f}%)")
        
        print(f"\nNumeric class vs Class name mapping:")
        mapping = test_df.groupby(['class', 'class name']).size().reset_index(name='count')
        print(mapping.to_string(index=False))
        
        print(f"\nUnique classes in Test: {sorted(test_df['class name'].unique())}")
    
    # ====================== TRAIN FILE ======================
    print("\n\nTRAIN FILE ANALYSIS")
    print("-" * 40)
    
    if not os.path.exists(train_path):
        print(f"Train file not found: {train_path}")
    else:
        train_df = pd.read_csv(train_path, header=None)
        print(f"Loaded train file: {train_df.shape[0]:,} annotations")
        
        class_col = 3
        image_col = 4
        
        print(f"Detected class column: {class_col} → '{train_df.iloc[0, class_col]}'")
        
        class_count_train = train_df[class_col].value_counts()
        print("\nClass Distribution (Train):")
        for cls, count in class_count_train.items():
            print(f"  {cls:12} : {count:6,}  ({count/len(train_df)*100:5.2f}%)")
        
        print(f"\nUnique classes in Train: {sorted(train_df[class_col].unique())}")
        
        print(f"\nUnique images in Train: {train_df[image_col].nunique():,}")
    
    print("\n\nCROSS-FILE COMPARISON")
    print("-" * 40)
    
    try:
        test_classes = set(test_df['class name'].unique())
        train_classes = set(train_df[class_col].unique())
        
        print(f"Classes only in Test : {test_classes - train_classes}")
        print(f"Classes only in Train: {train_classes - test_classes}")
        print(f"Common classes      : {test_classes & train_classes}")
        
    except NameError:
        print("Could not compare (one of the files failed to load)")
    
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")

if __name__ == "__main__":
    audit_csv_files(
        test_path="labels_test.csv",   
        train_path="labels_train.csv"
    )