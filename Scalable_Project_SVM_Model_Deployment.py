# This code performs data cleaning, feature engineering, and trains an SVM model on the Yelp Reviews data set.
#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Load Necessary Libraries
from pyspark.sql.functions import *
from pyspark.sql.functions import udf
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


# In[2]:


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF 
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col


sc = SparkContext("local[*]")
spark = SparkSession(sc)

#"review_id","user_id","business_id","stars","date","text","useful","funny","cool"

#1. Clean the dataset


# In[3]:

# Reading the Reviews Data Set
review = spark.read.option("header",True).csv( "yelp_review.csv", header=True)
review = review.withColumn("label", review["stars"].cast("double")).drop("stars")
review = review.dropna(subset=['label', 'text', 'funny', 'cool',"useful"])


# In[4]:


user = spark.read.option("header",True).csv( "yelp_user.csv", header=True)


# In[5]:


user1 = user.filter(~(user.elite == 'None'))
user2 = user1.select(['user_id']).withColumn("elite", lit(1))
user3 = user2.dropDuplicates()


# In[6]:


review1 = review.join(user3, on = 'user_id', how = 'left')


# In[7]:


review2 = review1.withColumn('elite', when(col('elite').isNull(), 0).otherwise(col('elite')))


# In[9]:

# Data Cleaning
# remove punctuation
def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    return nopunct

# binarize rating
def convert_rating(rating):
    rating = int(rating)
    if rating >=4: return 1
    else: return 0

# udf
punct_remover = udf(lambda x: remove_punct(x))
rating_convert = udf(lambda x: convert_rating(x))

# apply to review raw data
review_df = review2.select('review_id', punct_remover('text'), rating_convert('label'), "elite")

review_df = review_df.withColumnRenamed('<lambda>(text)', 'text').withColumn('label', review_df["<lambda>(label)"].cast(IntegerType())).drop('<lambda>(label)')

review_df.show(5)


# In[10]:


# tokenize
tok = Tokenizer(inputCol="text", outputCol="words")
review_tokenized = tok.transform(review_df)

# remove stop words
stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized = stopword_rm.transform(review_tokenized)


# In[14]:

# Saving Tokenizer Model to HDFS
tok.write().overwrite().save("tokenizer")


# In[15]:


from pyspark.sql.functions import *

review_tokenized = review_tokenized.withColumn('text', ltrim(review_tokenized.text))


# In[16]:


review_tokenized.show()


# In[17]:


# count vectorizer
cv = CountVectorizer(inputCol='words_nsw', outputCol='tf')
cvModel = cv.fit(review_tokenized)
count_vectorized = cvModel.transform(review_tokenized)


# In[18]:


# Saving Vectorizer Model to HDFS
cvModel.write().overwrite().save("cv_fit")


# In[19]:


count_vectorized.limit(10).show()


# In[20]:


from pyspark.ml.feature import IDF
idf = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidfModel = idf.fit(count_vectorized)
tfidf_df = tfidfModel.transform(count_vectorized)


# In[21]:


# Saving IDF Model to HDFS
tfidfModel.write().overwrite().save("tfidfModel")


# In[23]:


tfidf_df.select("elite").dropDuplicates().show()


# In[24]:


# split into training and testing set
# tfidf_df1 = tfidf_df.drop('text').select('tfidf','label')
tfidf_df1 = tfidf_df.drop('text').select('tfidf','label')
splits1 = tfidf_df1.randomSplit([0.8,0.2],seed=100)
train = splits1[0].cache()
test = splits1[1].cache()


# In[25]:


from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as MLLibVectors


# In[ ]:


tfidf_df1.select('label').dropDuplicates().show()


# In[ ]:


train.schema


# In[28]:


train_lb = train.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))


# In[29]:


# SVM model
numIterations = 50
regParam = 0.3
svm = SVMWithSGD.train(train_lb, numIterations, regParam=regParam)


# In[36]:


# Saving Trained SVM Model to HDFS
svm.save(sc, "svm_model")


# In[30]:


# predict
test_lb = test.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
scoreAndLabels_test = test_lb.map(lambda x: (float(svm.predict(x.features)), x.label))
score_label_test = spark.createDataFrame(scoreAndLabels_test, ["prediction", "label"])


# In[31]:


# F1 score
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
svm_f1 = f1_eval.evaluate(score_label_test)
print("F1 score: %.4f" % svm_f1)