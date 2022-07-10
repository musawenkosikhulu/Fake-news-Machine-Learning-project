#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #This library to read csv tables
import numpy as np #This library for mathematics
from sklearn.model_selection import train_test_split #This is for splitting data for training a ML model
from sklearn.metrics import classification_report #This is the model to report fake news
from sklearn.metrics import mean_absolute_error
import re #
import string  
import csv 
from sklearn.feature_extraction.text import TfidfVectorizer #This library is for vectoration of the words
from sklearn.linear_model import LogisticRegression #This code is for logistic regression model
from sklearn.tree import DecisionTreeClassifier #This code is for decision tree classifier.
from sklearn.ensemble import RandomForestClassifier #This code is for Random forest classifier


# In[2]:


fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')
true_news.head()
fake_news.head()


# In[3]:


true_news.shape, fake_news.shape  #Checking the shape of the data 


# In[4]:


true_news['class'] =1 #Creating classification for the true news
fake_news['class'] =0 #Creating classification for the fake news

true_news_handle = true_news.tail(20) #Taking only the last 20 data of the csv
for i in range(21416,213396,-1):
    true_news.drop([i],axis=0,inplace=True)
    
fake_news_handle = fake_news.tail(20) #Taking only the last 20 data of the csv
for i in range(23480,23460,-1):
    fake_news.drop([i],axis=0,inplace=True)


# In[5]:


handled_news = pd.concat([true_news_handle,fake_news_handle], axis=0)  #Combine the two dataset and change to cvd
handled_news.to_csv('Handled_news.csv')


# In[6]:


original_news_combined = pd.concat([true_news,fake_news],axis=0) #Combine the fake and true news into one dataset
original_news_combined = original_news_combined.sample(frac = 1) #This code sample the dataset


# In[7]:


original_news_combined.shape


# In[8]:


original_news_combined.isnull().sum() #This code checks if there is any null values in the dataset


# In[9]:


original_news_combined


# In[10]:


def remove_char(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[11]:


original_news_combined["text"] = original_news_combined["text"].apply(remove_char) #Removing unneccsesary charactor in the text


# In[12]:


original_news_combined["title"] = original_news_combined["title"].apply(remove_char) #Removing unneccsesary charactor in the title


# In[13]:


original_news_combined


# ### **Splitting of the data**

# In[14]:


X = original_news_combined["text"] #Creating the features we are going to perform the classificatio
Y = original_news_combined["class"] #This is the target feature


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,random_state=1) #Spliting the data for training and testing of the model


# ### **Vectorization of the text**

# In[16]:


vectorization = TfidfVectorizer() #This creates a function to turn the text to vectors
xv_train = vectorization.fit_transform(x_train) #Turning the trian data to vectors
xv_test = vectorization.transform(x_test) #Turning the test data to vectors


# In[17]:


xv_train.shape, y_train.shape #Checking if my data is still in shape for training


# ### **Logistic Regression**

# In[18]:


Logistic_Regression_model = LogisticRegression() #This code loads the regression model
Logistic_Regression_model.fit(xv_train,y_train) #This code fits the model
pred_lr=Logistic_Regression_model.predict(xv_test) #This code predicts the model
Logistic_Regression_model.score(xv_test, y_test) #This code right here give us the score of our model


# In[19]:


mean_absolute_error(y_test,pred_lr) #the magnitude of difference between the prediction of an observation and the true value of that observation


# ### **Precision – What percent of your predictions were correct?**
# Precision is the ability of a classifier not to label an instance positive that is actually negative. For each class it is defined as the ratio of true positives to the sum of true and false positives. [ref](https://muthu.co/understanding-the-classification-report-in-sklearn/)

# ### **Recall – What percent of the positive cases did you catch?** 
# 
# Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.  [ref](https://muthu.co/understanding-the-classification-report-in-sklearn/)

# ### **F1 score – What percent of positive predictions were correct?**
# 
# The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.  [ref](https://muthu.co/understanding-the-classification-report-in-sklearn/)

# In[20]:


print(classification_report(y_test, pred_lr)) #measure the quality of predictions from a classification algorithm


# ### **Decision Tree Classifier**

# In[21]:


Decision_Tree_model = DecisionTreeClassifier() #This code loads the decision tree model
Decision_Tree_model.fit(xv_train, y_train) #This code fits the model
pred_dt = Decision_Tree_model.predict(xv_test) #This code predicts the model
Decision_Tree_model.score(xv_test, y_test) #This code right here give us the score of our model


# In[22]:


mean_absolute_error(y_test,pred_dt) #the magnitude of difference between the prediction of an observation and the true value of that observation


# In[23]:


print(classification_report(y_test, pred_dt)) #measure the quality of predictions from a classification algorithm


# ### **Random Forest Classifier**

# In[24]:


Random_Forest_Classifier = RandomForestClassifier(random_state=0) #This code loads the random forest model
Random_Forest_Classifier.fit(xv_train, y_train) #This code fits the model
pred_Random_Forest_Classifier = Random_Forest_Classifier.predict(xv_test) #This code predicts the model
Random_Forest_Classifier.score(xv_test, y_test) #This code right here give us the score of our model


# In[25]:


mean_absolute_error(y_test,pred_Random_Forest_Classifier) #the magnitude of difference between the prediction of an observation and the true value of that observation


# In[26]:


print(classification_report(y_test, pred_Random_Forest_Classifier)) #measure the quality of predictions from a classification algorithm


# In[27]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def news_predictor(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(remove_char) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR=Logistic_Regression_model.predict(new_xv_test) #This code predicts the model
    mean_absolute_error(y_test,pred_LR) #the magnitude of difference between the prediction of an observation and the true value of that observation
    pred_DT = Decision_Tree_model.predict(new_xv_test) #This code predicts the model
    pred_RFC = Random_Forest_Classifier.predict(new_xv_test) #This code predicts the mode 
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0]),output_lable(pred_RFC[0])))


# In[28]:


news = str(input())
news_predictor(news)


# In[29]:


excel_data = pd.read_excel('Fake News (hinnews.com) Questionable.xlsx')


# In[31]:


excel_data


# In[ ]:




