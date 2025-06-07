from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create SparkSession
spark = SparkSession.builder.appName("Netflix ML Model").getOrCreate()

# Read clean data
df = spark.read.option("header", True).option("inferSchema", True).csv("data/netflix_cleaned.csv")

# Choose features and label
features = ["Open", "High", "Low", "Volume"]
label = "Close"

# Create columns
assembler = VectorAssembler(inputCols=features, outputCol="features")
data = assembler.transform(df).select("features", label)

# Split data train/test
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Create Linear Regression
lr = LinearRegression(featuresCol="features", labelCol=label)

# Train data
lr_model = lr.fit(train_data)

# Evaluate model
predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = lr_model.summary.r2

print(f"RMSE on test data: {rmse:.4f}")
print(f"R² on training data: {r2:.4f}")

# Print
print("Coefficients:", lr_model.coefficients)
print("Intercept:", lr_model.intercept)

spark.stop()
