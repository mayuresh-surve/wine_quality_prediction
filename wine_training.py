#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark


# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.rdd import reduce


# In[3]:


spark = SparkSession.builder.appName('WineQualityPredictionApp').getOrCreate()


# In[4]:


train_dataset= spark.read.format("com.databricks.spark.csv").csv(
    'TrainingDataset.csv', header=True, sep=";")
train_dataset.printSchema()


# In[5]:


train_dataset.show()


# In[6]:


validation_dataset= spark.read.format("com.databricks.spark.csv").csv(
    'ValidationDataset.csv', header=True, sep=";")
validation_dataset.printSchema()


# In[7]:


#To clean out CSV headers if quotes are present
old_train_dataset_column_name = train_dataset.schema.names
old_validation_dataset_column_name = validation_dataset.schema.names
clean_train_dataset_column_name = []
clean_validation_dataset_column_name = []

for name in old_train_dataset_column_name:
    clean_train_dataset_column_name.append(name.replace('"',''))
clean_train_dataset_column_name

for name in old_validation_dataset_column_name:
    clean_validation_dataset_column_name.append(name.replace('"',''))

clean_validation_dataset_column_name


# In[8]:


train_dataset = reduce(lambda train_dataset, idx: train_dataset.withColumnRenamed(old_train_dataset_column_name[idx], clean_train_dataset_column_name[idx]), range(len(clean_train_dataset_column_name)), train_dataset)
validation_dataset = reduce(lambda validation_dataset, idx: validation_dataset.withColumnRenamed(old_validation_dataset_column_name[idx], clean_validation_dataset_column_name[idx]), range(len(clean_validation_dataset_column_name)), validation_dataset)


# In[9]:


train_dataset=train_dataset.distinct()
validation_dataset=validation_dataset.distinct()


# In[10]:


total_columns = train_dataset.columns
tot_columns=validation_dataset.columns


# In[11]:


from pyspark.sql.functions import col
def preprocess(dataset):
    return dataset.select(*(col(c).cast("double").alias(c) for c in dataset.columns))
train_dataset = preprocess(train_dataset)
validation_dataset = preprocess(validation_dataset)


# In[12]:


from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
stages = []
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())


# In[13]:


for column_name in total_columns[:-1]:
    stages = []
    vectorAssembler = VectorAssembler(inputCols=[column_name],outputCol=column_name+'_vect')
    stages.append(vectorAssembler)
    stages.append(MinMaxScaler(inputCol=column_name+'_vect', outputCol=column_name+'_scaled'))
    pipeline = Pipeline(stages=stages)
    train_dataset = pipeline.fit(train_dataset).transform(train_dataset).withColumn(
        column_name+"_scaled", unlist(column_name+"_scaled")).drop(column_name+"_vect").drop(column_name)


# In[14]:


train_dataset.show(5)


# In[15]:


for column_name in total_columns[:-1]:
    stages = []
    vectorAssembler = VectorAssembler(inputCols=[column_name],outputCol=column_name+'_vect')
    stages.append(vectorAssembler)
    stages.append(MinMaxScaler(inputCol=column_name+'_vect', outputCol=column_name+'_scaled'))
    pipeline = Pipeline(stages=stages)
    validation_dataset = pipeline.fit(validation_dataset).transform(validation_dataset).withColumn(
        column_name+"_scaled", unlist(column_name+"_scaled")).drop(column_name+"_vect").drop(column_name)


# In[16]:


validation_dataset.show(5)


# In[17]:


vectorAssembler = VectorAssembler(
    inputCols=[column_name+"_scaled" for column_name in total_columns[:-1]],
    outputCol='features')


# In[18]:


from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer


# In[19]:


labelIndexer = StringIndexer(inputCol=total_columns[-1], outputCol="indexedLabel").fit(train_dataset)


rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol="features", numTrees=200)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, rf, labelConverter])


# In[20]:


model = pipeline.fit(train_dataset)


# In[21]:


model.write().overwrite().save("s3://winequalityprediction/models")


# In[ ]:


predictions = model.transform(validation_dataset)

# Select example rows to display.
predictions.select("predictedLabel", total_columns[-1], "features").show(5)


# In[ ]:


# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print("Accuracy is: ", accuracy)
print("F1 Score: ", f1_score)


# In[ ]:




