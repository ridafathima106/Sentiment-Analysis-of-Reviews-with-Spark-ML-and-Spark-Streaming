# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:01:55 2023

@author: Arumugam
"""
# Load Necessary Libraries
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType

from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import * 
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import string 
import re
from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
#from pyspark.ml.feature import Tokenizer, HashingTF, IDF 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql import Row
import numpy as np

# define a function to compute sentiments of the received tweets
def get_prediction(tweet_text):
    try:
        print('Started Preprocessing of Review')
        rowRdd = tweet_text.map(lambda w: Row(text=w))
        # create a spark dataframe
        df = spark.createDataFrame(rowRdd)

        df1 = df.withColumn("label", lit(0))
        # remove punctuation
        def remove_punct(text):
            regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
            nopunct = regex.sub(" ", text)
            nopunct1 = re.sub('\s+', ' ', nopunct).strip()
            return nopunct1
        
        # binarize rating
        def convert_rating(rating):
            rating = int(rating)
            if rating >=4: return 1
            else: return 0
        
        # udf
        punct_remover = udf(lambda x: remove_punct(x))
        rating_convert = udf(lambda x: convert_rating(x))
        
        # apply to review raw data
        df2 = df1.select(punct_remover('text'), "label")
        df2 = df2.withColumnRenamed('<lambda>(text)', 'text')
        df2 = df2.withColumn('text', ltrim(df2.text))
        tok = Tokenizer.load("tokenizer")
        df3 = tok.transform(df2)
        # remove stop words
        stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
        df4 = stopword_rm.transform(df3)
        cvModel = CountVectorizerModel.load("cv_fit")
        count_vectorized = cvModel.transform(df4)
        tfidfModel = IDFModel.load("tfidfModel")
        tfidf_df = tfidfModel.transform(count_vectorized)
        tfidf_df1 = tfidf_df.drop('text').select('tfidf','label')
        
        print('Start of Sentimental Analysis')

        # Load Trained Model from HDFS
        svm = SVMModel.load(sc, "svm_model")
        test_lb = tfidf_df1.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
        scoreAndLabels_test = test_lb.map(lambda x: (float(svm.predict(x.features)), x.label))
        score_label_test = spark.createDataFrame(scoreAndLabels_test, ["prediction", "label"])
        final_pred = score_label_test.first()['prediction']
        final_pred1 = np.where(final_pred == 1, 'Positive', 'Negative')
        print("The Review is ", final_pred1)
    except:
        print('No Data')

if __name__ == "__main__":
    sc = SparkContext("local[2]",appName = "NetworkWordCount")
    sc.setLogLevel("ERROR")
    ssc = StreamingContext(sc,30)
    spark = SparkSession(sc)
	
    #Create DStream from data source
    lines = ssc.socketTextStream("localhost",65395)

    #Transformations and actions on DStream
    #Split according to punctuation
    
    # To perform DF like operations        
    lines.foreachRDD(get_prediction)
    
    #Start listening to the server
    ssc.start()
    ssc.awaitTermination()
