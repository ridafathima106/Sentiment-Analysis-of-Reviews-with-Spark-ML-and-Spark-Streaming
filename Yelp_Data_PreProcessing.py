#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Loading Necessary Libraries
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


# In[7]:


# Read Review Data
review = spark.read.option("header",True).csv( "yelp_review.csv", header=True)


# In[ ]:


# Pre - Process Review Data
review = review.withColumn("label", review["stars"].cast("double"))
review = review.dropna(subset=['label', 'text', 'funny', 'cool',"useful"])
review = review.select(['review_id', 'user_id', 'business_id', 'stars', 'text'])


# In[9]:


# Read User Data
user = spark.read.option("header",True).csv( "yelp_user.csv", header=True)


# In[5]:


# Pre - Process User Data
user1 = user.select(['user_id', 'elite'])
user2 = user1.dropDuplicates()


# In[11]:


# Read Business Data
business = spark.read.option("header",True).csv("yelp_business.csv", header=True)


# In[12]:


# Pre - Process Business Data
business1 = business.select(["business_id", "state", "categories"]).dropDuplicates()


# In[13]:


review1 = review.join(user2, on = 'user_id', how = 'left')


# In[14]:


review2 = review1.join(business1, on = 'business_id', how = 'left')


# In[16]:


review2.columns


# In[17]:


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
review3 = review2.select('business_id','user_id','review_id','stars', punct_remover('text'), 'elite','state','categories')

review3 = review3.withColumnRenamed('<lambda>(text)', 'text')

review3.show(5)


# In[19]:


review3 = review3.withColumn('text', ltrim(review3.text))


# In[ ]:


#tok = Tokenizer.load("tokenizer")
#review4 = tok.transform(review3)
# remove stop words
#stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
#review5 = stopword_rm.transform(review4)
#print(review5.schema)
# In[ ]:

# Write Pre Processed Data to HDFS for EDA purposes
review3.write.format("csv").mode("overwrite").option("head", "true").save("PreProcessedFile.csv")


