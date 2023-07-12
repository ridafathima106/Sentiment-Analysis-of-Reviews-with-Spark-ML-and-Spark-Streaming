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


review3 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("C:/Users/arumu/PreProcessedDataSet/*.csv")


# In[12]:


from pyspark.ml.feature import StopWordsRemover

# tokenize
tok = Tokenizer(inputCol="text", outputCol="words")
review_tokenized = tok.transform(review3.filter(review3.text.isNotNull()))

# create a list of custom stop words
custom_stopwords = ['one','s']

# create the StopWordsRemover with the custom stop words
stopword_rm = StopWordsRemover(inputCol='words', outputCol='words_nsw', stopWords=custom_stopwords)

# apply the StopWordsRemover to your DataFrame
review_tokenized = stopword_rm.transform(review_tokenized)


# In[13]:


from pyspark.sql.functions import *

review_tokenized = review_tokenized.withColumn('text', ltrim(review_tokenized.text))


# # EDA 1 - Word Cloud

# In[15]:


#combining all rows into single list

reviews_words=review_tokenized.limit(50000).toPandas()
reviews_words['words_nsw']
combined_list = reviews_words['words_nsw'].sum()


# In[21]:


#word cloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# convert list to string
text = ' '.join(combined_list)

# create a WordCloud object
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)

# display the word cloud
plt.figure(figsize=(20, 20), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# # Word Count Frequency Analysis

# In[16]:


review_tokenized1 = review_tokenized.withColumn("Word_Count",size(review_tokenized.words_nsw))


# In[17]:


from pyspark.sql.functions import col, split, size
import matplotlib.pyplot as plt

# create a DataFrame containing positive reviews
positive_df = review_tokenized1.select("Word_Count","stars").filter(col("stars").isin([4, 5]))

# create a DataFrame containing negative reviews
negative_df = review_tokenized1.select("Word_Count","stars").filter(col("stars").isin([1,2,3]))


# In[18]:


bins, hist = positive_df.select(["Word_Count"]).limit(180000).rdd.flatMap(lambda x: x).histogram(10)


# In[19]:


bins1, hist1 = negative_df.select(["Word_Count"]).limit(180000).rdd.flatMap(lambda x: x).histogram(10)


# In[20]:


plt.hist(bins[:-1], bins=bins, weights=hist, color="Teal")
plt.title("Word Count for Positive Reviews")
plt.xlabel("Number Of Words")
plt.ylabel("Frequency")



plt.show()


# In[21]:


plt.hist(bins[:-1], bins=bins1, weights=hist1, color="Maroon")
plt.title("Word Count for Negative Reviews")
plt.xlabel("Number Of Words")
plt.ylabel("Frequency")

plt.show()


# # EDA3

# In[22]:


review4 = review3.withColumn("categories", explode(split("categories", ";")))
# Split the DataFrame into two based on the value of the "rating" column
positive_reviews = review4.filter(review3.stars >= 4)
negative_reviews = review4.filter(review3.stars < 4)


# In[23]:


positive_review_counts = (positive_reviews.filter(~positive_reviews.categories.isin(['0','1'])).groupBy("Categories").count()).orderBy("count", ascending=False)
#positive_review_counts.show(10)
positive_limit = (positive_review_counts.limit(10)).toPandas()
positive_limit


# In[24]:


import matplotlib.pyplot as plt

# sort the dataframe by the count column in ascending order
positive_limit_sorted = positive_limit.sort_values(by="count")

plt.figure(figsize=(12,6))
plt.barh(positive_limit_sorted["Categories"], positive_limit_sorted["count"], color="orange")
plt.xlabel("Business Category")
plt.ylabel("Number of Positive reviews")
plt.title("Businesses with top positive reviews")
plt.show()


# In[25]:


negative_review_counts = (negative_reviews.filter(~negative_reviews.categories.isin(['0','1'])).groupBy("Categories").count()).orderBy("count", ascending=False)
negative_limit = (negative_review_counts.limit(10)).toPandas()
negative_limit


# In[26]:


import matplotlib.pyplot as plt

# sort the dataframe by the count column in ascending order
negative_limit_sorted = negative_limit.sort_values(by="count")

plt.figure(figsize=(12,6))
plt.barh(negative_limit_sorted["Categories"], negative_limit_sorted["count"], color="Red")
plt.xlabel("Business Category")
plt.ylabel("Number of Negative reviews")
plt.title("Businesses with top Negative reviews")
plt.show()


# # EDA4

# In[44]:


user


# In[45]:


import plotly.graph_objects as go

# count the number of elite users and common users
elite_count = user.filter(user.elite != 'None').count()
common_count = user.filter(user.elite == 'None').count()

# create a bar chart using Plotly
fig = go.Figure(go.Bar(x=['Elite Users', 'Common Users'], y=[elite_count, common_count]))

# set the plot title and axis labels
fig.update_layout(title='Number of Elite Users vs. Common Users', xaxis_title='User Type', yaxis_title='User Count')

# remove 0 from y-axis label
fig.update_layout(yaxis_tickformat="")

# show the plot
fig.show()


# In[20]:


e = review3.filter(review3["elite"] != "None")
ne = review3.filter(review3["elite"] == "None")


# In[21]:


elite_df = e.select("user_id", "review_id", "stars")
nonelite_df = ne.select("user_id", "review_id", "stars")


# In[22]:


from pyspark.sql.functions import count
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp

# Group by stars and get the count for "e" DataFrame
e_count_df = elite_df.groupBy("stars").agg(count("*").alias("count")).toPandas()
e_count_df = e_count_df.sort_values(by="stars")

# Create a bar plot for "e" DataFrame
fig1 = go.Figure([go.Bar(x=e_count_df["stars"], y=e_count_df["count"])])

# Group by stars and get the count for "ne" DataFrame
ne_count_df = nonelite_df.groupBy("stars").agg(count("*").alias("count")).toPandas()
ne_count_df = ne_count_df.sort_values(by="stars")

# Create a bar plot for "ne" DataFrame
fig2 = go.Figure([go.Bar(x=ne_count_df["stars"], y=ne_count_df["count"])])

# Create a subplot to show both plots side by side
fig = sp.make_subplots(rows=1, cols=2, subplot_titles=("Elite Users", "Non-Elite Users"))

# Add the plots to the subplot
fig.add_trace(fig1.data[0], row=1, col=1)
fig.add_trace(fig2.data[0], row=1, col=2)

# Set the layout for the subplot
fig.update_layout(title="Distribution of Stars by User Type", xaxis_title="Stars", yaxis_title="Count")

# Show the plot
fig.show()


# In[17]:


review3


# In[27]:


review3=review3.filter(review3.stars<=5)


# In[28]:


eda_o1=review3.groupby('stars').agg(count("*").alias("count")).toPandas()


# In[32]:


# import necessary libraries
from pyspark.sql import SparkSession
import plotly.graph_objects as go
from pyspark.sql.functions import count

# create a bar chart using Plotly
fig = go.Figure(go.Bar(x=eda_o1["stars"], y=eda_o1["count"]))

# arrange the x-axis in 1,2,3,4,5 form
fig.update_layout(xaxis=dict(type="category", categoryorder="array", categoryarray=[1, 2, 3, 4, 5]))

# set the x and y axis labels
fig.update_layout(xaxis_title="Stars", yaxis_title="Count")

# remove 0 from y-axis label
fig.update_layout(yaxis_tickformat="")

# show the plot
fig.show()

# stop the SparkSession
spark.stop()


# In[23]:


review3 = spark.read.option("header",True).csv( "yelp_review.csv", header=True)

review3 = review3.withColumn("label", review3["stars"].cast("double"))
review3 = review3.dropna(subset=['label', 'text', 'funny', 'cool',"useful"])
review3 = review3.filter(review3.label<=5)


# In[24]:


review3


# In[ ]:


from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# create a udf to apply the lambda function
sia = SentimentIntensityAnalyzer()
sia_udf = udf(lambda text: sia.polarity_scores(text)['compound'], FloatType())

# create a new column 'sentiment' using the udf
review3 = review3.select('*', sia_udf('text').alias('sentiment'))

# plot a histogram of tip sentiment
sns.histplot(review3.select('sentiment').toPandas())


# In[ ]:




