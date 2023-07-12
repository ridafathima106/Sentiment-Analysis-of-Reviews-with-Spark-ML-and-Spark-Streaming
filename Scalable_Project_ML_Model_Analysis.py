#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


review = spark.read.option("header",True).csv( "hdfs://user/spark/yelp_review.csv", header=True)
review = review.withColumn("label", review["stars"].cast("double")).drop("stars")
review = review.dropna(subset=['label', 'text', 'funny', 'cool',"useful"])


# In[4]:


# remove punctuation
def remove_punct(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    nopunct = regex.sub(" ", text)
    return nopunct

#remove extra spaces
def remove_spaces(text):
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    clean_text = regex.sub(" ", text)
    clean_text = re.sub(r'\s{2,}', ' ', clean_text)  # replace patterns with two or more spaces with a single space
    return clean_text


# binarize rating
def convert_rating(rating):
    rating = int(rating)
    if rating >=4: return 1
    else: return 0

# udf
punct_remover = udf(lambda x: remove_punct(x))
rating_convert = udf(lambda x: convert_rating(x))
remove_spaces_udf = udf(lambda x: remove_spaces(x))

# apply to review raw data
review_df = review.select('review_id', remove_spaces_udf(punct_remover('text')), rating_convert('label'))


review_df = review_df.withColumnRenamed('<lambda>(<lambda>(text))', 'text')                     .withColumn('label', review_df["<lambda>(label)"].cast(IntegerType()))                     .drop('<lambda>(label)')


review_df.show(5)


# In[5]:


# tokenize
tok = Tokenizer(inputCol="text", outputCol="words")
review_tokenized = tok.transform(review_df)

# remove stop words
stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw')
review_tokenized = stopword_rm.transform(review_tokenized)


# In[6]:


from pyspark.sql.functions import *

review_tokenized = review_tokenized.withColumn('text', ltrim(review_tokenized.text))


# In[7]:


review_tokenized.limit(20).toPandas()


# In[8]:


#combining all rows into single list

reviews_1=review_tokenized.toPandas()
reviews_1['words_nsw']
combined_list = reviews_1['words_nsw'].sum()


# In[ ]:


combined_list


#


# count vectorizer
cv = CountVectorizer(inputCol='words_nsw', outputCol='tf')
cvModel = cv.fit(review_tokenized)
count_vectorized = cvModel.transform(review_tokenized)


# In[ ]:


count_vectorized.limit(10).show()


# In[ ]:


from pyspark.ml.feature import IDF
idf = IDF().setInputCol('tf').setOutputCol('tfidf')
tfidfModel = idf.fit(count_vectorized)
tfidf_df = tfidfModel.transform(count_vectorized)


# In[ ]:


tfidf_df.show()


# In[ ]:


# split into training and testing set
tfidf_df1 = tfidf_df.drop('text').select('tfidf','label').filter(tfidf_df.label != 2017)
splits1 = tfidf_df1.randomSplit([0.8,0.2],seed=100)
train = splits1[0].cache()
test = splits1[1].cache()


# In[ ]:


from pyspark.mllib.classification import SVMModel, SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as MLLibVectors


# In[ ]:


tfidf_df1.select('label').dropDuplicates().show()


# In[ ]:


# convert to LabeledPoint vectors
train_lb = train.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
test_lb = test.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))


# In[ ]:


train.schema


# In[ ]:


# SVM model
numIterations = 50
regParam = 0.3
svm = SVMWithSGD.train(train_lb, numIterations, regParam=regParam)


# In[ ]:


# predict
test_lb = test.rdd.map(lambda row: LabeledPoint(row[1], MLLibVectors.fromML(row[0])))
scoreAndLabels_test = test_lb.map(lambda x: (float(svm.predict(x.features)), x.label))
score_label_test = spark.createDataFrame(scoreAndLabels_test, ["prediction", "label"])


# In[ ]:


# F1 score
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
svm_f1 = f1_eval.evaluate(score_label_test)
print("F1 score: %.4f" % svm_f1)


# In[ ]:


vocabulary = cvModel.vocabulary
weights = svm.weights.toArray()
svm_coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})


# In[ ]:


svm_coeffs_df.sort_values('weight').head(5)


# In[ ]:


#Logistic Regression


# In[ ]:


from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
numIterations = 10
regParam = 0.3
logistic = LogisticRegressionWithLBFGS.train(train_lb, iterations=numIterations, regParam=regParam)


# In[ ]:


# evaluate on test set
logreg_preds = test_lb.map(lambda x: (float(logistic.predict(x.features)), x.label))
logreg_metrics = spark.createDataFrame(logreg_preds, ["prediction", "label"])


# In[ ]:


# F1 score
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
svm_f1 = f1_eval.evaluate(logreg_metrics)
print("F1 score: %.4f" % svm_f1)


# In[ ]:


vocabulary = sorted(set(cvModel.vocabulary))
weights = logistic.weights.toArray()
logistic_coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights[:len(vocabulary)]})


# In[ ]:


logistic_coeffs_df.sort_values('weight').head(5)


# In[ ]:


from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Train a Naive Bayes model
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
nb_model = nb.fit(train_lb)

# Make predictions on test data
nb_preds = nb_model.transform(test_lb)

# Evaluate the model using F1 score
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
nb_f1 = f1_eval.evaluate(nb_preds)
print("F1 score: %.4f" % nb_f1)

# Get the vocabulary and weights of the model
vocabulary = cv_model.vocabulary
weights = nb_model.coefficients.toArray()
nb_coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})


# In[ ]:


from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint

# Train a Naive Bayes model
nb_model = NaiveBayes.train(train_lb, 1.0)


# In[ ]:


# Make predictions on test data
nb_preds = test_lb.map(lambda p: (nb_model.predict(p.features), p.label))


# In[ ]:


# Convert nb_preds to a list
nb_preds_list = nb_preds.collect()
nb_preds_list = [(float(x), float(y)) for x, y in nb_preds_list]

# Create DataFrame from nb_preds_list
nb_metrics = spark.createDataFrame(nb_preds_list, ["prediction", "label"])
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
nb_f1 = f1_eval.evaluate(nb_metrics)
print("F1 score: %.4f" % nb_f1)


# In[ ]:




