import sys
import os
import random
import time
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
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,RegressionEvaluator,BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer,VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier

from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector, Vectors, DenseVector
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import MulticlassMetrics,RegressionMetrics,BinaryClassificationMetrics

# parse semi-colon delimited file to a LabeledPoint object without header and column 1 as label (rest as feature)
def regParsePoint(line):  
	  values = [float(x) for x in line.split(';')]
	  return LabeledPoint(float(values[0]), values[1:])
     
def classParsePoint(line):  
	  values = [x for x in line.split(';')]
	  return LabeledPoint(values[0], values[1:])     

def remove_folder(path):
	# check if folder exists
	if os.path.exists(path):
		# remove if exists
		shutil.rmtree(path)   

class SPARK_MODEL: 
      
  #init model params    
  def __init__(self, dataset,dataName,splitRatio,targetType,targetVariable,split,nbSamples,goodClass,sparkModelsId,sparkLearningMethods,sparkOptions,numClasses, extDataSet):             
    self.dataset = dataset
    self.dataName = dataName
    self.splitRatio = splitRatio
    self.targetType = targetType
    self.targetVariable = targetVariable
    self.split = split    
    self.nbSamples = nbSamples
    self.goodClass = goodClass
    self.sparkModelsId = sparkModelsId
    self.sparkLearningMethods = sparkLearningMethods
    self.sparkOptions = sparkOptions
    self.numClasses = numClasses
    self.extDataSet = extDataSet         

  #rdd methods
  def _set_rdd(self, dataset):  
    self._rdd = sc.textFile(dataset, 8)  
    header = self._rdd.first() 
    self._rdd = self._rdd.filter(lambda line:line != header)
    
    if self.targetType == 'classification':
      print "class"
      self._rdd = self._rdd.map(classParsePoint)
    else:
      self._rdd = self._rdd.map(regParsePoint)      
         
    print self._rdd.first()                

  def _get_rdd(self):
    return self._rdd    

  def _get_rddTest(self):
    return self._rddTest

  def _get_rddTraining(self):
    return self._rddTraining 
  
  def _get_rddModel(self):
    return self._rddModel

  #model building: rdd        
  def _set_rddModel(self, _type, _SLA, data):
    if _type == 'regression':
      if _SLA ==  'randomForest':
        self._rddModel = RandomForest.trainRegressor(data, categoricalFeaturesInfo={},numTrees=int(self.sparkOptions[4]), featureSubsetStrategy=self.sparkOptions[5],impurity='variance',           maxDepth=int(self.sparkOptions[1]), maxBins=32) 
      else:
        self._rddModel = ""
    else:    #classification
      if _SLA ==  'randomForest':
        print self.numClasses
        self._rddModel = RandomForest.trainClassifier(data, numClasses=self.numClasses,          categoricalFeaturesInfo={},numTrees=int(self.sparkOptions[4]),                                             maxDepth=int(self.sparkOptions[1]), featureSubsetStrategy=self.sparkOptions[5],impurity=self.sparkOptions[2])           
      else:
        self._rddModel = ""
        
  def splitData(self):
    if self.split != "ExternalValidation":
      (self._rddTest, self._rddTraining) = self._rdd.randomSplit([1-self.splitRatio, self.splitRatio])
    else:
    
      print "ExternalValidation"
      self._rddTraining = self._rdd
      
      self._rddTest = sc.textFile(self.extDataSet, 8)  
      header = self._rddTest.first() 
      self._rddTest = self._rddTest.filter(lambda line:line != header)
    
      if self.targetType == 'classification':      
        self._rddTest = self._rddTest.map(classParsePoint)
      else:
        self._rddTest = self._rddTest.map(regParsePoint)                   
   
  
  #rdd/dataFrame method
  def rddToDataFrame(self,rdd):
    return rdd.toDF()  
  
  def dataFrameToRdd(self,dataFrame):
    return dataFrame.rdd   
  
  #dataFrame method
  def _set_dataFrame(self):
    self._dataFrame = sqlContext.read.format('csv').options(delimiter=';', header='true',inferschema='true',nullValue='').load(self.dataset)    
    self._dataFrame = self._dataFrame.withColumn(self.targetVariable, self.dataFrame[self.targetVariable].cast("double"))  

  def _get_dataFrame(self):
    return self._dataFrame    

  def _get_dataFrameTest(self):
    return self._dataFrameTest

  def _get_dataFrameTraining(self):
    return self._dataFrameTraining    

  def splitDataFrameData(self):
    if self.split != "ExternalValidation":
      (self._rddTest, self._rddTraining) = self.dataFrameToRdd(self._get_dataFrame()).randomSplit([1-self.splitRatio, self.splitRatio])       
    else:     
      self.splitData()
      
    self._dataFrameTest = self._rddTest.toDF()
    self._dataFrameTraining = self._rddTraining.toDF()
    
  def _get_dataFrameModel(self):
    return self._dataFrameModel
  
  def _get_pipeline(self):
    return self._pipeline
  
  def _get_crossval(self):
    return self._crossval
    
  def _get_paramGrid(self):
    return self._paramGrid      
    
  def _get_regEval(self):
    return self._regEval    
  
  #model building: dataframe
  def _set_dataFrameModel(self, _type, _SLA, data,vecAssembler):     

    if _type == 'regression':
      if _SLA ==  'randomForest':
        rf = RandomForestRegressor()
        rf.setLabelCol(self.targetVariable)\
          .setPredictionCol("prediction")\
          .setFeaturesCol("features")\
          .setProbabilityCol("proba")\
          .setSeed(100088121L)\
          .setMaxDepth(int(self.sparkOptions[1]))\
          .setMaxMemoryInMB(10000)\
          .setFeatureSubsetStrategy(self.sparkOptions[5])          
        self._regEval = RegressionEvaluator(predictionCol="prediction", labelCol=self.targetVariable, metricName="rmse")  

    else:    #classification
      if _SLA ==  'randomForest':        
        rf = RandomForestClassifier(labelCol=self.targetVariable, featuresCol="features", maxDepth=int(self.sparkOptions[1]),featureSubsetStrategy = self.sparkOptions[5],impurity = self.sparkOptions[2],probabilityCol="proba")        
        if goodClass != '':
          self.regEval = BinaryClassificationEvaluator(labelCol=self.targetVariable, metricName="areaUnderROC")
        else:
          self.regEval = MulticlassClassificationEvaluator(labelCol=self.targetVariable, predictionCol="prediction", metricName="accuracy")    
        
    # Create a Pipeline
    self._pipeline = Pipeline()    
    # Set the stages of the Pipeline #vecAssembler
    self._pipeline.setStages([vecAssembler,rf])
    # GridSearch
    self._paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [int(num) for num in self.sparkOptions[4].split(',')] )
                 .build())      
    # Add the grid to the CrossValidator
    self._crossval = CrossValidator(estimator=self._pipeline,
                              estimatorParamMaps=self._paramGrid,
                              evaluator=self._regEval,
                              numFolds=self.nbSamples)        
    # Now let's find and return the best model
    self._dataFrameModel = self._crossval.fit(data).bestModel
    
    #to be removed
    #print rf.getNumTrees()    
    #modelText = str(self._dataFrameModel.stages[-1])    
    #._java_obj.toDebugString()    
    #nbTrees = int(re.sub('.*?([0-9]*) trees$',r'\1',modelText))
    #print nbTrees
    # end TBR        
    
    rf.save("/home/t752887/python/myModelPath/SPARK_RF_R_"+ str(self.sparkModelsId[0]))
    
  #end function
          
  #model evaluation
  #classification
  def computeKappa(self,m):
  		
  		sum = np.sum(m)
  	
  		row = m.sum(axis=0)
  		col = m.sum(axis=1)
  	
  		P0 = m.trace()/sum
      		
  		PE = np.sum((row[i]/sum)*(col[i]/sum) for i in range(m.shape[0]))
  		return (P0 - PE)/(1 - PE)
  
  def computeBA(self,m):
  		row = m.sum(axis=0)
  		col = m.sum(axis=1)
  		return np.sum( m[i][i]/col[i] for i in range(m.shape[0]) )/m.shape[0]    
    
  #rdd model evalution  
  def getRddPredictionsLabels(self,model, test_data):
    predictions = model.predict(test_data.map(lambda r: r.features))
    return predictions.zip(test_data.map(lambda r: r.label))
  
  def printRddMulticlassClassificationMetrics(self, predictions_and_labels):
    metrics = MulticlassMetrics(predictions_and_labels)
    print "KAPPA="+str(self.computeKappa(np.array(metrics.confusionMatrix().toArray())))
    print "BA="+str(self.computeBA(np.array(metrics.confusionMatrix().toArray())))
    CMarray = metrics.confusionMatrix().toArray()
    #CMstring = ','.join(['%.5f' % num for num in CMarray])
    print "CM="+str(CMarray)
    
  def printRddBinaryClassificationMetrics(self, predictions_and_labels):
    metrics = BinaryClassificationMetrics(predictions_and_labels)
    print "KAPPA="+str(self.computeKappa(np.array(metrics.confusionMatrix().toArray())))
    print "BA="+str(self.computeBA(np.array(metrics.confusionMatrix().toArray())))
    CMarray = metrics.confusionMatrix().toArray()
    #CMstring = ','.join(['%.5f' % num for num in CMarray])
    print "CM="+str(CMarray)    
    
  def evaluateRddClassificationModel(self):  
      predictions_and_labels = self.getRddPredictionsLabels(self._get_rddModel(), self._get_rddTest())
      if self.goodClass != '': #binary classification
        #self.printRddBinaryClassificationMetrics(predictions_and_labels)
        self.printRddMulticlassClassificationMetrics(predictions_and_labels)
      else:
        self.printRddMulticlassClassificationMetrics(predictions_and_labels)    
    
  def evaluateRddRegressionModel(self):
      # Get predictions
      valuesAndPreds = self.getRddPredictionsLabels(self._get_rddModel(), self._get_rddTest())
      # Instantiate metrics object
      metrics = RegressionMetrics(valuesAndPreds)    
      # Squared Error
      print("MSE = %s" % metrics.meanSquaredError)
      print("RMSE = %s" % metrics.rootMeanSquaredError)    
      # R-squared
      print("R-squared = %s" % metrics.r2)    
      # Mean absolute error
      print("MAE = %s" % metrics.meanAbsoluteError)    
      # Explained variance
      print("Explained variance = %s" % metrics.explainedVariance)    

  def evaluateDataFrameRegressionModel(self):
      # Now let's use rfModel to compute an evaluation metric for our test dataset: testSetDF
      predictionsAndLabelsDF = self._dataFrameModel.transform(self._dataFrameTest)
              
      # Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
      rmseRF = self._regEval.evaluate(predictionsAndLabelsDF)
      
      # Now let's compute the r2 evaluation metric for our test dataset
      r2RF = self._regEval.evaluate(predictionsAndLabelsDF, {self._regEval.metricName: "r2"})
      
      print("RMSE = %s" % rmseRF)
      print("R-squared = %s " % r2RF)      
      
  def evaluateDataFrameClassificationModel(self, sc):
    #here we have a problem
    a = 1

  #save models
  def saveRddModel(self, sc):
    #save rdd API model
    remove_folder("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+self.sparkModelsId[0]) 
    modelPath = "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+ str(self.sparkModelsId[0]);
    self._rddModel.save(sc, modelPath)

  def saveDataFrameModel(self):
      #final model to save  
      #self._dataFrameModel = self._pipeline.fit(self._dataFrame)
      self._dataFrameModel = self._crossval.fit(self._dataFrame).bestModel
      
      modelText = str(self._dataFrameModel.stages[-1])    
      #._java_obj.toDebugString()    
      nbTrees = int(re.sub('.*?([0-9]*) trees$',r'\1',modelText))
      print nbTrees      
      
      #save data frame API model
      remove_folder("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+self.sparkModelsId[0])      
      modelPath = "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+ str(self.sparkModelsId[0]);
      self._dataFrameModel.save(modelPath)
      self._pipeline.save(modelPath+"_Pipeline")      

        
  def buildRDDModel(self, sparkContext):
  
    print "RDD_MODEL"
  
    # init RDD from dataset
    self._set_rdd(self.dataset)    
    # split into test - training set
    self.splitData()     
    # save rddTest and rddTraining into CSV and copy to PLP server!

    #self._rddTest.toDF().write.csv('/home/t752887/data/output/'+self.sparkModelsId[0]+'_'+self.dataName+'_test.csv')
    
    #self._rddTraining.toDF().write.csv('/home/t752887/data/output/'+self.sparkModelsId[0]+'_'+self.dataName+'_training.csv')    
    self._rddTraining.toDF().toPandas().to_csv('/home/t752887/data/output/'+self.sparkModelsId[0]+'_'+self.dataName+'_training.csv')
    
    self._rddTest.toDF().toPandas().to_csv('/home/t752887/data/output/'+self.sparkModelsId[0]+'_'+self.dataName+'_test.csv')
    
    #lines = self._rddTest.map(toCSVLine)
    #lines.saveAsTextFile('/home/t752887/data/output/'+self.sparkModelsId[0]+'_'+self.dataName+'_test.csv')
    
    #lines = self._rddTraining.map(toCSVLine)
    #lines.saveAsTextFile('/home/t752887/data/output/'+self.sparkModelsId[0]+'_'+self.dataName+'_training.csv')

    #could become a loop of models
    if self.targetType == 'classification':
      self._set_rddModel('classification','randomForest',self._get_rddTraining())
      
      self.evaluateRddClassificationModel()

      #final model to save
      self._set_rddModel('classification','randomForest',self._get_rdd())        
        
    #regression
    else:
      self._set_rddModel('regression','randomForest',self._get_rddTraining())    
 
      self.evaluateRddRegressionModel()
      
      #final model to save
      self._set_rddModel('regression','randomForest',self._get_rdd())      
  
    #TODO: save the model
    self.saveRddModel(sparkContext)

  def buildDataFrameModel(self): 
    # init dataframe from dataset   
    self._set_dataFrame()
    # split into test - training set
    self.splitDataFrameData()
    
    #vector assembler
    ignore = [self.targetVariable]
    vecAssembler = VectorAssembler(inputCols=[x for x in self._dataFrameTraining.columns if x not in ignore], outputCol="features")
  
    #dataFrame cross-validation Pipeline with model selection
    if self.targetType == 'regression':
      #build model on the data we pass            
      self._set_dataFrameModel('regression','randomForest',self._get_dataFrameTraining(),vecAssembler) 
      #evaluate best model
      self.evaluateDataFrameRegressionModel()            
      # save the model
      self.saveDataFrameModel()

    else:
      #build model on the data we pass            
      self._set_dataFrameModel('regression','randomForest',self._get_dataFrameTraining(),vecAssembler) 
      #TODO evaluate best model
      self.evaluateDataFrameClassificationModel(sparkContext)            
      #TODO save the model
      self.saveDataFrameModel(sparkContext)    



  def performModelSelection(self):
    try:
      i = float(self.sparkOptions[4])
      return 0
    except (ValueError, TypeError):
      return 1
    
  dataFrame = property(_get_dataFrame, _set_dataFrame)
  dataFrameTest = property(_get_dataFrameTest) 
  dataFrameTraining = property(_get_dataFrameTraining)
  dataFrameModel = property(_get_dataFrameModel, _set_dataFrameModel)
  pipeline = property(_get_pipeline)
  crossval = property(_get_crossval)
  paramGrid = property(_get_paramGrid) 
  regEval = property(_get_regEval)
    
  rdd = property(_get_rdd, _set_rdd)
  rddTest = property(_get_rddTest)    
  rddTraining = property(_get_rddTraining)   
  rddModel = property(_get_rddModel, _set_rddModel)  
        
#init Spark
conf = SparkConf().setAppName("RFC").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)          

print str(sys.argv[1]).replace("'", "")
print str(sys.argv[6]).replace("'", "")
print str(sys.argv[7]).replace("'", "")
print str(sys.argv[8]).replace("'", "")
print str(sys.argv[9]).replace("'", "")

#init new model
model = SPARK_MODEL(str(sys.argv[1]).replace("'", ""),
                    str(sys.argv[2]).replace("'", ""),
                    float(str(sys.argv[4]).replace("'", "")),
                    str(sys.argv[5]).replace("'", ""),
                    str(sys.argv[6]).replace("'", ""),
                    str(sys.argv[7]).replace("'", ""),
                    int(str(sys.argv[9]).replace("'", "")),
                    str(sys.argv[11]).replace("'", ""),
                    eval(sys.argv[12]),
                    eval(sys.argv[13]),
                    eval(sys.argv[14]),
                    int(str(sys.argv[15]).replace("'", "")),
                    str(sys.argv[16]).replace("'", ""),
                    )

print model.sparkOptions[0] #minimum sample per node (=min_sample_leaf) -> can be tuned 
print model.sparkOptions[1] #tree depth
print model.sparkOptions[2] #split method (Gini/Entropy)
print model.sparkOptions[3] #weightingMethod (Uniform/ByClass/Bagging?)
print model.sparkOptions[4] #nbtree 
print model.sparkOptions[5] #number of descriptors/features to consider at each split (Default,All,Sqrt,Ln)-> can be tuned

if model.performModelSelection():
  model.buildDataFrameModel()
else:  
  model.buildRDDModel(sc)
  


'''

def remove_folder(path):
	# check if folder exists
	if os.path.exists(path):
		# remove if exists
		shutil.rmtree(path)   
				
			
#parse each line of data as a dense vector and a label?
def vectorizeData(data):
	return data.map(lambda r: [str(r[0]), Vectors.dense(r[1:])]).toDF(['label','features'])
	#vectorized_CV_data = vectorizeData(CV_data)


a = datetime.now()


#no model selection
try:
  i = float(SLA_OPT[4])
 
  rdd = sc.textFile(dataset, 8)
  header = rdd.first() 
  rdd = rdd.filter(lambda line:line != header)
  rdd = rdd.map(parsePoint)             
  (rddTest, rddTraining) = rdd.randomSplit([1-splitRatio, splitRatio])  
  
  #regression 10k * 512 - 50t;30d: 84s (57s Spark)
  if targetType == 'regression':
    model = RandomForest.trainRegressor(rddTraining, categoricalFeaturesInfo={},
                                      numTrees=int(SLA_OPT[4]), featureSubsetStrategy="auto",
                                      impurity='variance', maxDepth=int(SLA_OPT[1]), maxBins=32)
    # Get predictions
    valuesAndPreds = getPredictionsLabels(model, rddTest)
    # Instantiate metrics object
    metrics = RegressionMetrics(valuesAndPreds)    
    # Squared Error
    print("MSE = %s" % metrics.meanSquaredError)
    print("RMSE = %s" % metrics.rootMeanSquaredError)    
    # R-squared
    print("R-squared = %s" % metrics.r2)    
    # Mean absolute error
    print("MAE = %s" % metrics.meanAbsoluteError)    
    # Explained variance
    print("Explained variance = %s" % metrics.explainedVariance)    
    
    #final model to save
    model = RandomForest.trainRegressor(rdd, categoricalFeaturesInfo={},
                                  numTrees=int(SLA_OPT[4]), featureSubsetStrategy="auto",
                                  impurity='variance', maxDepth=int(SLA_OPT[1]), maxBins=32)
    
  #classification -> binary or multiclass?? (more important for model selection) 
  #binary classification 6k * 512 - 50t;30d:   
  else: 
    model = RandomForest.trainClassifier(rddTraining, numClasses=nbClasses, categoricalFeaturesInfo={},numTrees=int(SLA_OPT[4]),maxDepth=int(SLA_OPT[1]), featureSubsetStrategy="auto")
    predictions_and_labels = getPredictionsLabels(model, rddTest)
    printClassificationMetrics(predictions_and_labels)
  
    #final model to save  
    model = RandomForest.trainClassifier(rdd, numClasses=nbClasses, categoricalFeaturesInfo={},numTrees=int(SLA_OPT[4]),maxDepth=int(SLA_OPT[1]), featureSubsetStrategy="auto")
  
  #save rdd API model
  remove_folder("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+modelIds[0]) 
  modelPath = "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+ str(modelIds[0]);
  model.save(sc, modelPath)
  
except (ValueError, TypeError):
  
  datasetDF = sqlContext.read.format('csv').options(delimiter=';', header='true',inferschema='true',nullValue='').load(dataset)  
  datasetDF = datasetDF.withColumn(targetVariable, datasetDF[targetVariable].cast("double"))
  
  (split20DF, split80DF) = datasetDF.rdd.randomSplit([1-splitRatio, splitRatio])
  testSetDF = split20DF.toDF()
  trainingSetDF = split80DF.toDF()  

  ignore = [targetVariable]
  vecAssembler = VectorAssembler(inputCols=[x for x in trainingSetDF.columns if x not in ignore], outputCol="features")

    
  #regression 10k * 512 - (50,1)t;30d (7 models):   516s
  if targetType == 'regression':
    rf = RandomForestRegressor()
    rf.setLabelCol(targetVariable)\
      .setPredictionCol("prediction")\
      .setFeaturesCol("features")\
      .setSeed(100088121L)\
      .setMaxDepth(int(SLA_OPT[1]))\
      .setMaxMemoryInMB(10000)
    regEval = RegressionEvaluator(predictionCol="prediction", labelCol=targetVariable, metricName="rmse")    
  else:
    rf = RandomForestClassifier(labelCol=targetVariable, featuresCol="features", maxDepth=int(SLA_OPT[1]))
    if goodClass != '':
      regEval = BinaryClassificationEvaluator(labelCol=targetVariable, metricName="areaUnderROC")
    else:
      regEval = MulticlassClassificationEvaluator(labelCol=targetVariable, predictionCol="prediction", metricName="accuracy")    

  #.setNumTrees(int(SLA_OPT[4]))\
  # Create a Pipeline
  rfPipeline = Pipeline()
  
  # Set the stages of the Pipeline #vecAssembler
  rfPipeline.setStages([vecAssembler,rf])
  
  paramGrid = (ParamGridBuilder()
               .addGrid(rf.numTrees, [int(num) for num in SLA_OPT[4].split(',')] )
               .build())
  
  # Add the grid to the CrossValidator
  crossval = CrossValidator(estimator=rfPipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=regEval,
                            numFolds=folds)  
  
  # Now let's find and return the best model
  rfModel = crossval.fit(trainingSetDF).bestModel
    
  
  # Now let's use rfModel to compute an evaluation metric for our test dataset: testSetDF
  predictionsAndLabelsDF = rfModel.transform(testSetDF)
      
  if targetType == 'regression':
    # Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
    rmseRF = regEval.evaluate(predictionsAndLabelsDF)
    
    # Now let's compute the r2 evaluation metric for our test dataset
    r2RF = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})
    
    print("RF RMSE: {0:.2f}".format(rmseRF))
    print("RF Q2: {0:.2f}".format(r2RF))
    
    #final model to save  
    #model = pipeline.fit(trainingData)
      
  else:
    #areaUnderROC = regEval.evaluate(predictionsAndLabelsDF)
    #print("RF Accuracy: {0:.2f}".format(areaUnderROC))
    
    #myRdd = datasetDF.rdd.map(parsePoint)
    rdd = sc.textFile(dataset, 8)
    header = rdd.first() 
    rdd = rdd.filter(lambda line:line != header)
    rdd = rdd.map(parsePoint)             
    (rddTest, rddTraining) = rdd.randomSplit([0.8, 0.2])    
    
    modelText = str(rfModel.stages[-1])
    
    #._java_obj.toDebugString()    
    nbTrees = int(re.sub('.*?([0-9]*) trees$',r'\1',modelText))
    
    print nbTrees
    
    model = RandomForest.trainClassifier(rddTraining,categoricalFeaturesInfo={}, numClasses=nbClasses,numTrees=nbTrees,maxDepth=int(SLA_OPT[1]),featureSubsetStrategy="auto")
    predictions_and_labels = getPredictionsLabels(model, rddTest)
    printClassificationMetrics(predictions_and_labels)
    
    #final model to save  
    model = RandomForest.trainClassifier(rdd, categoricalFeaturesInfo={}, numClasses=nbClasses,numTrees=int(nbTrees),maxDepth=int(SLA_OPT[1]), featureSubsetStrategy="auto")

  #save data frame API model
  remove_folder("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+modelIds[0])
#sparkModel.getModel().save(sc, "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+modelIds[0]) 
  modelPath = "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+ str(modelIds[0]);
  rfModel.save(modelPath)
  rfPipeline.save(modelPath+"_Pipeline")



b = datetime.now()

print str((b-a).total_seconds()) + " seconds"
'''

'''
#7.7s for 9.6k rows * 0.8k col (6.5s for 10rows)
#18.3s for 284k rows * 3.2k col
datasetDF = sqlContext.read.format('csv').options(delimiter=';', header='true',inferschema='true',nullValue='').load(dataset)

datasetDF = datasetDF.withColumn(targetVariable, datasetDF[targetVariable].cast("double"))

#datasetDF.cache()

(split20DF, split80DF) = datasetDF.rdd.randomSplit([1-splitRatio, splitRatio])
testSetDF = split20DF.toDF()
trainingSetDF = split80DF.toDF()

ignore = [targetVariable]
#indexer = VectorIndexer(maxCategories=2, inputCol=[x for x in datasetDF.columns if x not in ignore], outputCol="indexed")
#model = indexer.fit(datasetDF)

#vecAssembler = VectorAssembler(inputCols=[x for x in testSetDF.columns if x not in ignore], outputCol="features")

vecAssembler = VectorAssembler(inputCols=[x for x in trainingSetDF.columns if x not in ignore], outputCol="features")

#testSetDF = vecAssembler.transform(testSetDF)
#trainingSetDF = vecAssembler.transform(trainingSetDF)

#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(datasetDF)

#datasetDF = featureIndexer.transform(datasetDF)

#selector = ChiSqSelector(numTopFeatures=50, featuresCol="features", outputCol="selectedFeatures", labelCol="Physchem*logP*Physchem").fit(datasetDF)

#datasetDF = selector.transform(datasetDF)

#newDataSet = datasetDF.select("selectedFeatures")

# Create a RandomForestRegressor
if targetType == 'regression':
  print "regression"
  rf = RandomForestRegressor()
  rf.setLabelCol(targetVariable)\
    .setPredictionCol("prediction")\
    .setFeaturesCol("features")\
    .setSeed(100088121L)\
    .setMaxDepth(int(SLA_OPT[1]))\
    .setNumTrees(int(SLA_OPT[4]))\
    .setMaxMemoryInMB(10000)
  regEval = RegressionEvaluator(predictionCol="prediction", labelCol=targetVariable, metricName="rmse")
else:
  print "classification"
  rf = RandomForestClassifier(labelCol=targetVariable, featuresCol="features", numTrees=10)
  #regEval = MulticlassClassificationEvaluator(labelCol=targetVariable, predictionCol="prediction", metricName="accuracy")
  regEval = BinaryClassificationEvaluator(labelCol=targetVariable, metricName="areaUnderROC")


# Create a Pipeline
rfPipeline = Pipeline()

# Set the stages of the Pipeline #vecAssembler
rfPipeline.setStages([vecAssembler,rf])

#crossval = CrossValidator(estimator=rfPipeline, evaluator=regEval, numFolds=folds)

#crossval.setEstimator(rfPipeline)

# Let's tune over our rf.maxBins parameter on the values 50 and 100, create a parameter grid using the ParamGridBuilder
paramGrid = (ParamGridBuilder()
            # .addGrid(rf.maxBins, [30,50,100,200])
             .build())

# Add the grid to the CrossValidator
#crossval.setEstimatorParamMaps(paramGrid)
crossval = CrossValidator(estimator=rfPipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=regEval,
                          numFolds=folds)


#print trainingSetDF.take(5)
# Now let's find and return the best model
fitRes = crossval.fit(trainingSetDF)

rfModel = fitRes.bestModel

print "type: "+str(type(rfModel))


# Now let's use rfModel to compute an evaluation metric for our test dataset: testSetDF
predictionsAndLabelsDF = rfModel.transform(testSetDF)



if targetType == 'regression':
  # Run the previously created RMSE evaluator, regEval, on the predictionsAndLabelsDF DataFrame
  rmseRF = regEval.evaluate(predictionsAndLabelsDF)
  
  # Now let's compute the r2 evaluation metric for our test dataset
  r2RF = regEval.evaluate(predictionsAndLabelsDF, {regEval.metricName: "r2"})
  
  print("RF RMSE: {0:.2f}".format(rmseRF))
  print("RF Q2: {0:.2f}".format(r2RF))
else:
  areaUnderROC = regEval.evaluate(predictionsAndLabelsDF)
  print("RF Accuracy: {0:.2f}".format(areaUnderROC))
  
  # use spark_RF_CV3.py -> maybe need to skip first row!
  print datasetDF.count()
  
  #myRdd = datasetDF.rdd.map(parsePoint)
  rdd = sc.textFile(dataset, 8)
  header = rdd.first() 
  rdd = rdd.filter(lambda line:line != header)
  rdd = rdd.map(parsePoint)             
  (rddTest, rddTraining) = rdd.randomSplit([0.8, 0.2])    
  
  model = RandomForest.trainClassifier(rddTraining, numClasses=2, categoricalFeaturesInfo={},numTrees=5,maxDepth=3, featureSubsetStrategy="auto")
  predictions_and_labels = getPredictionsLabels(model, rddTest)
  printClassificationMetrics(predictions_and_labels)

b = datetime.now()

print str((b-a).total_seconds()) + " seconds"

#create rdd
#rdd = data.map(parsePoint)
#cache rdd
#rdd.cache()

#always split in training/test set
#training_data, testing_data = rdd.randomSplit([0.8, 0.2])

#no model selection 
#sparkModel.setModel(RandomForest.trainClassifier(training_data, numClasses=9, categoricalFeaturesInfo={},
#                                     numTrees=50,maxDepth=30, featureSubsetStrategy="auto",
#                                     impurity='gini',  maxBins=32))                                     


#predictions_and_labels = sparkModel.getPredictionsLabels(testing_data)

#sparkModel.printClassificationMetrics(sparkModel.getPredictionsLabels(testing_data))

remove_folder("/home/t752887/python/myModelPath/SPARK_RF_Regression_"+modelIds[0])
#sparkModel.getModel().save(sc, "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+modelIds[0])

modelPath = "/home/t752887/python/myModelPath/SPARK_RF_Regression_"+ str(modelIds[0]);
rfModel.save(modelPath)
rfPipeline.save(modelPath+"_Pipeline")

#training_data, testing_data = rdd.randomSplit([0.8, 0.2])

#no model selection -> we could do that after model selection

#sparkModel.setModel(RandomForest.trainClassifier(split80DF, numClasses=2, categoricalFeaturesInfo={},
#                                     numTrees=50,maxDepth=30, featureSubsetStrategy="auto",
#                                     impurity='gini',  maxBins=32))                                     


#predictions_and_labels = sparkModel.getPredictionsLabels(split20DF)

#sparkModel.printClassificationMetrics(sparkModel.getPredictionsLabels(split20DF))

#predictions
#rfModel.load(modelPath)
#rfPipeline.load(modelPath+"_Pipeline")


#datasetDF = sqlContext.read.format('csv').options(delimiter=';', header='true',inferschema='true',nullValue='').load(dataset)

#(split20DF, split80DF) = datasetDF.rdd.randomSplit([0.01,0.99])
#testSetDF = split20DF.toDF()

#predictions = rfModel.transform(testSetDF)
# Select example rows to display.
predictions.select("prediction", targetVariable, "features").show(5)
#print modelPath
#rfModel.save(sc, modelPath)
#what we need to return is the confidence metrics and that's it? (the best model if model selection)
#maybe also some raw files
'''
