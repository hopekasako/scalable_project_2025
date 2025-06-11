from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col,
    lag,
    round as spark_round,
    mean,
    when,
    row_number,
    stddev,
    abs,
    greatest,
    sum as spark_sum,
    dayofweek,
    month,
    quarter,
    try_divide,
)

spark = SparkSession.builder.appName("Netflix Data Featuring").getOrCreate()

df = spark.read.csv(path="data/nflx_cleaned.csv", header=True, inferSchema=True)
df.printSchema()

# Shorting data by Date
df = df.orderBy(col("Date"))

# Define window specifications
window_spec = Window.orderBy(col("Date"))
window_spec_5 = Window.orderBy(col("Date")).rowsBetween(-4, 0)  # 5-day window
window_spec_9 = Window.orderBy(col("Date")).rowsBetween(-8, 0)  # 9-day window
window_spec_10 = Window.orderBy(col("Date")).rowsBetween(-9, 0)  # 10-day window
window_spec_12 = Window.orderBy(col("Date")).rowsBetween(-11, 0)  # 12-day window
window_spec_14 = Window.orderBy(col("Date")).rowsBetween(-13, 0)  # 14-day window
window_spec_20 = Window.orderBy(col("Date")).rowsBetween(-19, 0)  # 20-day window
window_spec_26 = Window.orderBy(col("Date")).rowsBetween(-25, 0)  # 25-day window


# 1. Price-based features
df = df.withColumn("Daily_Range", col("High") - col("Low"))
df = df.withColumn("Price_Change", col("Close") - col("Open"))
df = df.withColumn(
    "Pct_change", spark_round((col("Close") - col("Open")) / col("Open"), 6)
)

# 2. Previous Close and Returns
df = df.withColumn("Prev_Close", lag(col("Close")).over(window_spec))
df = df.withColumn(
    "Returns", spark_round((col("Close") - col("Prev_Close")) / col("Prev_Close"), 6)
)

# 3. Moving Averages
df = df.withColumn("SMA_5", spark_round(mean(col("Close")).over(window_spec_5), 2))
df = df.withColumn("SMA_10", spark_round(mean(col("Close")).over(window_spec_10), 2))
df = df.withColumn("SMA_20", spark_round(mean(col("Close")).over(window_spec_20), 2))


# EMA calculation
def calculate_ema(df, column, window_size, window_spec):
    alpha = 2 / (window_size + 1)
    df = df.withColumn(f"EMA_{window_size}_temp", col(column))
    for i in range(1, window_size):
        df = df.withColumn(
            f"EMA_{window_size}_temp",
            when(
                row_number().over(window_spec) > i,
                (alpha * col(column))
                + (
                    (1 - alpha)
                    * lag(col(f"EMA_{window_size}_temp"), 1).over(window_spec)
                ),
            ).otherwise(col(f"EMA_{window_size}_temp")),
        )
    return df.withColumn(
        f"EMA_{window_size}", spark_round(col(f"EMA_{window_size}_temp"), 2)
    ).drop(f"EMA_{window_size}_temp")


df = calculate_ema(df, "Close", 12, window_spec)
df = calculate_ema(df, "Close", 26, window_spec)

# 4. MACD
df = df.withColumn("MACD", col("EMA_12") - col("EMA_26"))
df = df.withColumn("Signal_Line", spark_round(mean(col("MACD")).over(window_spec_9), 2))
df = df.withColumn("MACD_Histogram", col("MACD") - col("Signal_Line"))

# 5. RSI
df = df.withColumn("Price_Diff", col("Close") - lag(col("Close"), 1).over(window_spec))
df = df.withColumn("Gain", when(col("Price_Diff") > 0, col("Price_Diff")).otherwise(0))
df = df.withColumn("Loss", when(col("Price_Diff") < 0, -col("Price_Diff")).otherwise(0))
df = df.withColumn("Avg_Gain", spark_round(mean(col("Gain")).over(window_spec_14), 2))
df = df.withColumn("Avg_Loss", spark_round(mean(col("Loss")).over(window_spec_14), 2))
df = df.withColumn("RS", try_divide(col("Avg_Gain"), col("Avg_Loss")))
df = df.withColumn("RSI", spark_round(100 - (100 / (1 + col("RS"))), 2))

# 6. Bollinger Bands
df = df.withColumn("Std_20", spark_round(stddev(col("Close")).over(window_spec_20), 2))
df = df.withColumn("Upper_BB", col("SMA_20") + 2 * col("Std_20"))
df = df.withColumn("Lower_BB", col("SMA_20") - 2 * col("Std_20"))

# 7. Average True Range (ATR)
df = df.withColumn("TR1", col("High") - col("Low"))
df = df.withColumn("TR2", abs(col("High") - lag(col("Close"), 1).over(window_spec)))
df = df.withColumn("TR3", abs(col("Low") - lag(col("Close"), 1).over(window_spec)))
df = df.withColumn("TR", greatest(col("TR1"), col("TR2"), col("TR3")))
df = df.withColumn("ATR", spark_round(mean(col("TR")).over(window_spec_14), 2))

# 8. Volume-based features
df = df.withColumn("Prev_Volume", lag(col("Volume")).over(window_spec))
df = df.withColumn(
    "Volume_Pct_Change",
    spark_round((col("Volume") - col("Prev_Volume")) / col("Prev_Volume"), 6),
)

# On-Balance Volume (OBV)
df = df.withColumn(
    "OBV",
    spark_sum(
        when(col("Close") > lag(col("Close"), 1).over(window_spec), col("Volume"))
        .when(col("Close") < lag(col("Close"), 1).over(window_spec), -col("Volume"))
        .otherwise(0)
    ).over(window_spec),
)

# 9. Lagged Closing Prices
df = df.withColumn("Lag_Close_1", lag(col("Close"), 1).over(window_spec))
df = df.withColumn("Lag_Close_2", lag(col("Close"), 2).over(window_spec))
df = df.withColumn("Lag_Close_3", lag(col("Close"), 3).over(window_spec))

# 10. Date-based features
df = df.withColumn("Day_of_Week", dayofweek(col("Date")))
df = df.withColumn("Month", month(col("Date")))
df = df.withColumn("Quarter", quarter(col("Date")))

# Drop intermediate columns
df = df.drop("Price_Diff", "Gain", "Loss", "RS", "TR1", "TR2", "TR3", "TR", "Std_20")
df = df.withColumnRenamed("Adj Close", "Adj_Close")

# Save the output
df.write.csv("data/nflx_features.csv", header=True, mode="overwrite")

spark.stop()
