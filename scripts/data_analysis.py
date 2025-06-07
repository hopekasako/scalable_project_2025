from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, avg, max, min

spark = SparkSession.builder.appName("Netflix Data Analysis").getOrCreate()

# read data
df = spark.read.option("header", True).option("inferSchema", True).csv("data/netflix_cleaned.csv")

# Avg closed up to year
df = df.withColumn("Year", year("Date"))

avg_close_by_year = df.groupBy("Year").agg(avg("Close").alias("Avg_Close")).orderBy("Year")
avg_close_by_year.show()

# Max - Min Volume
df.select(max("Volume").alias("Max Volume"), min("Volume").alias("Min Volume")).show()

spark.stop()
