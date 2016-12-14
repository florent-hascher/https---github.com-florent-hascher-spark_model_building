import sys
import os
import random
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import re

from pyspark import SparkContext, StorageLevel, SparkConf
from pyspark.sql import SQLContext,DataFrame
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer,VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier


# Load and parse the features
def parseFeatures(line):
    #values = [float(x) for x in line.split(';')]
    values = [float(x.replace("[", "").replace("]", "")) for x in line.split(';')]
    return values
    
# Load and parse the features
def parseTarget(line):
    #values = [float(x) for x in line.split(';')]
    values = [float(x.replace("[", "").replace("]", "")) for x in line.split(';')]
    return values[0]  

conf = SparkConf().setAppName("RFC").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

dataId = str(sys.argv[1]).replace("'", "")

#print dataId

dataType = str(sys.argv[2]).replace("'", "")
targetName = str(sys.argv[3]).replace("'", "")
IS_RDD = str(sys.argv[4]).replace("'", "")
#print targetName
#we could pass also if reg or class or biclass and RDD or DataFrame must be found somewhere

#we should have a variable indicating if is RDD or DataFrame and what type of model
if IS_RDD:  
  rfModel = RandomForestModel.load(sc, "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+dataId)
  data = sc.textFile("/home/t752887/data/PRED_DATASET.csv")
  parsedFeaturesData = data.map(parseFeatures)  
  #parsedTargetData = data.map(parseTarget)  
  #predict them all!!
  print "prediction"
  print rfModel.predict(parsedFeaturesData).collect()  

else:  
  rfPipeline = Pipeline()
  #rfModel = CrossValidatorModel()  
  rfPipeline.load("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+dataId+"_Pipeline")
  rfModel = PipelineModel.load("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+dataId)
  rf = RandomForestRegressor.load("/home/t752887/python/myModelPath/SPARK_RF_R_"+dataId)
  #input should be a similar file as used for building the model with only one row per compound to predict (and of course no response)
  datasetDF = sqlContext.read.format('csv').options(delimiter=';', header='true',inferschema='true',nullValue='').load("/home/t752887/data/PRED_DATASET.csv")
  
  #print rf.getNumTrees()
  #modelText = str(rfModel.stages[-1].params)
  #print modelText
  #nbTrees = int(re.sub('.*?([0-9]*) trees$',r'\1',modelText)) 
  #print nbTrees
  
  predictions = rfModel.transform(datasetDF).select("prediction")
  
  print predictions.toPandas().to_string(index=False)
  