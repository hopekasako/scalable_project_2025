#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_outlier_detection(df, output_dir='distribution_plots'):
    """
    Perform detailed outlier detection on all numeric columns of the DataFrame.
    Saves a combined boxplot of all numeric columns to output_dir/outliers_boxplot.png
    Returns a dict of outlier info per column.
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    total_count = len(df)

    plt.figure(figsize=(15, 6))
    outlier_results = {}

    for idx, col in enumerate(numeric_columns, start=1):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers = df[mask]

        outlier_results[col] = {
            'total_outliers': len(outliers),
            'percentage_outliers': len(outliers) / total_count * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outliers
        }

        print(f"\nOutlier Analysis for {col}:")
        print(f"  Total Outliers: {len(outliers)}")
        print(f"  Percentage of Outliers: {len(outliers) / total_count * 100:.2f}%")
        print(f"  Lower Bound: {lower_bound:.4f}")
        print(f"  Upper Bound: {upper_bound:.4f}")
        if len(outliers) > 0:
            print("  Sample Outliers:")
            print(outliers.head().to_string())

        plt.subplot(1, len(numeric_columns), idx)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outliers_boxplot.png'))
    plt.close()
    return outlier_results

def main():
    # Setup
    CSV_INPUT = 'scripts/dataset/NFLX.csv'
    PREPROCESSED_OUTPUT = 'scripts/dataset/preprocessed_exp.csv'
    PLOTS_DIR = 'distribution_plots'

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Read dataset
    df = pd.read_csv(CSV_INPUT)
    print(f"Number of Rows: {df.shape[0]}")
    print(f"Number of Columns: {df.shape[1]}")
    print("\nColumn Data Types:")
    print(df.dtypes.to_string())

    # Outlier detection
    print("\nPerforming outlier detection...")
    outliers = comprehensive_outlier_detection(df, PLOTS_DIR)

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum().to_string())

    # Duplicate records
    dup_mask = df.duplicated()
    print(f"\nNumber of Duplicates: {dup_mask.sum()}")
    if dup_mask.any():
        print("Duplicate rows:")
        print(df[dup_mask].to_string())

    # Convert Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Interpolate missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].interpolate()

    print("\nPreprocessing complete:")
    print("- Date column converted to datetime (if existed)")
    print("- Numeric missing values interpolated")

    # Save processed data
    df.to_csv(PREPROCESSED_OUTPUT, index=False)
    print(f"\nPreprocessed data saved to '{PREPROCESSED_OUTPUT}'")

    # Descriptive statistics
    print("\nDescriptive statistics for numeric columns:")
    print(df[numeric_cols].describe().to_string())

    # Distribution histograms
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(2, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'histograms.png'))
    plt.close()

    # Boxplots
    plt.figure(figsize=(15, 5))
    df[numeric_cols].plot(kind='box')
    plt.title('Box Plot of Numeric Columns')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'boxplots.png'))
    plt.close()

    # Correlation heatmap
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))
    plt.close()

    print("\nCorrelation matrix:")
    print(corr.to_string())

    # Preview processed data
    print("\nProcessed Data Preview (first 5 rows):")
    print(df.head().to_string())

if __name__ == '__main__':
    main()