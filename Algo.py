#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle


# In[5]:


data = pd.read_csv("comments.csv",encoding='utf-8')


# In[6]:


print(data)


# In[8]:


dt=pd.DataFrame(data)
clean=dt['CONTENT'].str.replace('[^a-zA-Z\s]+|X{2,}', ' ')


# In[9]:


print(dt)


# In[10]:


dt.head()


# In[11]:


vectorizer = CountVectorizer()
count = vectorizer.fit_transform(dt['CONTENT'].values)

classifier = MultinomialNB()
target = dt['CLASS'].values
classifier.fit(count,target)


# In[12]:


pickle.dump(classifier,open('model.pkl','wb'))


# In[13]:


model = pickle.load(open('model.pkl','rb'))


# In[14]:


count=['follow my channels plzz']
example_count=vectorizer.transform(count)
predictions=classifier.predict(example_count)


# In[15]:


print(count,predictions)


# In[ ]:




