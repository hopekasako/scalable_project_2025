from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, expr, desc, sqrt, mean, stddev, percentile_approx, last, coalesce
from pyspark.sql.types import DoubleType, DateType
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("NetflixStockPreprocessing") \
    .getOrCreate()

# Create directories for plots and output if they don't exist
os.makedirs('distribution_plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

def comprehensive_outlier_detection(spark_df, numeric_columns):
    outlier_results = {}

    # Convert to pandas for visualization
    pandas_df = spark_df.toPandas()

    # Create a figure for boxplots
    plt.figure(figsize=(15, 6))

    total_count = spark_df.count()

    # Analyze each numeric column
    for idx, column in enumerate(numeric_columns, 1):
        # Calculate statistics using Spark
        stats = spark_df.select(
            mean(col(column)).alias('mean'),
            stddev(col(column)).alias('stddev'),
            expr(f'percentile_approx({column}, array(0.25, 0.75), 10000)').alias('quartiles')
        ).collect()[0]

        # Extract values
        q1, q3 = float(stats['quartiles'][0]), float(stats['quartiles'][1])
        iqr = q3 - q1

        # Define bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Count outliers using Spark
        outliers_count = spark_df.filter(
            (col(column) < lower_bound) | (col(column) > upper_bound)
        ).count()

        # Get outlier rows as pandas
        outliers = spark_df.filter(
            (col(column) < lower_bound) | (col(column) > upper_bound)
        ).toPandas()

        # Store results
        outlier_results[column] = {
            'total_outliers': outliers_count,
            'percentage_outliers': (outliers_count / total_count) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outliers
        }

        # Print detailed outlier information
        print(f"\nOutlier Analysis for {column}:")
        print(f"  Total Outliers: {outliers_count}")
        print(f"  Percentage of Outliers: {(outliers_count / total_count) * 100:.2f}%")
        print(f"  Lower Bound: {lower_bound:.4f}")
        print(f"  Upper Bound: {upper_bound:.4f}")

        if outliers_count > 0:
            print("  Sample Outliers:")
            print(outliers.head())

        # Create boxplot
        plt.subplot(1, len(numeric_columns), idx)
        sns.boxplot(x=pandas_df[column])
        plt.title(f'Boxplot of {column}')

    # Save boxplot
    plt.tight_layout()
    plt.savefig('distribution_plots/outliers_boxplot.png')
    plt.close()

    return outlier_results

def check_missing_values(df):
    total_count = df.count()
    for column in df.columns:
        missing_count = df.filter(col(column).isNull()).count()
        print(f"{column}: {missing_count} missing values ({(missing_count/total_count)*100:.2f}%)")

# Read the CSV file from 'data' folder

df = spark.read.csv("data/NFLX.csv", header=True, inferSchema=True)

# Display basic information
print(f"Number of Rows: {df.count()}")
print(f"Number of Columns: {len(df.columns)}")
print("\nColumn Data Types:")
for field in df.schema.fields:
    print(f"  {field.name}: {field.dataType}")

# OUTLIER DETECTION
numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
print("\nPerforming Outlier Detection...")
outlier_analysis = comprehensive_outlier_detection(df, numeric_columns)

# MISSING VALUES
print("\nMissing Values:")
check_missing_values(df)

# DATA TYPE CONVERSIONS
df = df.withColumn("Date", col("Date").cast(DateType()))
for column in numeric_columns:
    df = df.withColumn(column, col(column).cast(DoubleType()))

# FORWARD FILL MISSING VALUES
window_spec = Window.orderBy("Date")
for column in numeric_columns:
    df = df.withColumn(column,
                       coalesce(col(column),
                                last(col(column), True).over(window_spec)))

# SUMMARY STATISTICS
from pyspark.sql.functions import min as spark_min, max as spark_max
stats_list = []
for c in numeric_columns:
    stats_list += [
        mean(col(c)).alias(f"{c}_mean"),
        stddev(col(c)).alias(f"{c}_stddev"),
        spark_min(col(c)).alias(f"{c}_min"),
        spark_max(col(c)).alias(f"{c}_max")
    ]
summary_stats = df.select(stats_list).toPandas()
print("\nDescriptive Statistics:")
print(summary_stats)

# DISTRIBUTION PLOTS
pandas_df = df.toPandas()
plt.figure(figsize=(15, 10))
for i, col_name in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=pandas_df, x=col_name, kde=True)
    plt.title(f'{col_name} Distribution')
plt.tight_layout()
plt.savefig('distribution_plots/histograms.png')
plt.close()

plt.figure(figsize=(15, 5))
pandas_df[numeric_columns].boxplot()
plt.title('Box Plot of Numeric Columns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('distribution_plots/boxplots.png')
plt.close()

# CORRELATION HEATMAP
correlations = []
for col1 in numeric_columns:
    for col2 in numeric_columns:
        corr_value = df.stat.corr(col1, col2)
        correlations.append((col1, col2, corr_value))
correlation_df = pd.DataFrame(correlations, columns=['Column1', 'Column2', 'Correlation'])
correlation_matrix = correlation_df.pivot(index='Column1', columns='Column2', values='Correlation')

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('distribution_plots/correlation_heatmap.png')
plt.close()

# Save processed data back to 'data' folder
df.write.mode("overwrite").csv("data/preprocessed_spark_data.csv", header=True)
print("\nPreprocessed data saved to 'data/preprocessed_spark_data.csv'")

# Stop Spark session
spark.stop()