# %%
import pyspark
import argparse
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import PipelineModel, Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler 
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.rdd import reduce

# %%
spark = SparkSession.builder.appName('WineApp_Prediction').getOrCreate()

# %%
def read_csv(file_path):
    return spark.read.format("com.databricks.spark.csv").csv(
        file_path, header=True, sep=";")

# %%
def load_model(path):
    return PipelineModel.load(path) # for random forest

# %%
def preprocess(df):
    df = df.select(*(col(c).cast("double").alias(c) for c in df.columns))

    stages = []
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())
    
    old_train_dataset_column_name = df.schema.names
    clean_train_dataset_column_name = []

    for name in old_train_dataset_column_name:
        clean_train_dataset_column_name.append(name.replace('"',''))
    # print(clean_train_dataset_column_name)
    df = reduce(lambda df, idx: df.withColumnRenamed(old_train_dataset_column_name[idx], clean_train_dataset_column_name[idx]), range(len(clean_train_dataset_column_name)), df)
    
    total_columns = df.columns
    
    for column_name in total_columns[:-1]:
        stages = []
        vectorAssembler = VectorAssembler(inputCols=[column_name],outputCol=column_name+'_vect')
        stages.append(vectorAssembler)
        stages.append(MinMaxScaler(inputCol=column_name+'_vect', outputCol=column_name+'_scaled'))
        pipeline = Pipeline(stages=stages)
        df = pipeline.fit(df).transform(df).withColumn(
            column_name+"_scaled", unlist(column_name+"_scaled")).drop(
            column_name+"_vect").drop(column_name)
    return df, total_columns

# %%
def get_predictions(model, df):
    return model.transform(df)

# %%
def run(test_file):
    df = read_csv(test_file)
    df, total_columns = preprocess(df)
    model = load_model("s3://winequalityprediction/models/")
    df = get_predictions(model, df)
    return df, total_columns

# %%
def print_f1(df):
    evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction")
    accuracy = evaluator.evaluate(df, {evaluator.metricName: "accuracy"})
    f1_score = evaluator.evaluate(df, {evaluator.metricName: "f1"})

    print("Accuracy is: ", accuracy)
    print("F1 Score: ", f1_score)
    

# %%
import argparse
parser = argparse.ArgumentParser(description='Wine Quality prediction')
parser.add_argument('--test_file', required=True, help='please provide test file path you can provide s3 path or local file path')
args = parser.parse_args()
df, total_columns = run(args.test_file)
print_f1(df)

# %%



