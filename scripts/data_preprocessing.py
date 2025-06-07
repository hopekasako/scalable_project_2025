from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.sql.types import DoubleType

# Create SparkSession
spark = SparkSession.builder \
    .appName("Netflix Data Preprocessing") \
    .getOrCreate()

# Read data
df = spark.read.option("header", True).csv("data/NFLX.csv")

# Check original schema
print("Schema:")
df.printSchema()

# Edit type data và date and number column
df = df.withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
       .withColumn("Open", col("Open").cast(DoubleType())) \
       .withColumn("High", col("High").cast(DoubleType())) \
       .withColumn("Low", col("Low").cast(DoubleType())) \
       .withColumn("Close", col("Close").cast(DoubleType())) \
       .withColumn("Adj Close", col("Adj Close").cast(DoubleType())) \
       .withColumn("Volume", col("Volume").cast("long"))

# Delete null rows
df_cleaned = df.dropna()

# Drop for duplicates
df_cleaned = df_cleaned.dropDuplicates()

# Check data again
print("Data after clean:")
df_cleaned.show(5)
df_cleaned.printSchema()

# Export cleaned data into csv file
df_cleaned.coalesce(1).write.option("header", True).csv("data/netflix_cleaned.csv", mode="overwrite")

# Stop SparkSession
spark.stop()
