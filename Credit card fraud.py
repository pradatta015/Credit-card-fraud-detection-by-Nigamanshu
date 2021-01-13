#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('python: {}'.format(sys.version))
print('numpy: {}'.format(numpy.__version__))
print('pandas: {}'.format(pandas.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('scipy: {}'.format(scipy.__version__))
print('sklearn: {}'.format(sklearn.__version__))


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


#load data set
data = pd.read_csv('creditcard.csv')


# In[10]:


#explore data set
print(data.columns)


# In[11]:


print(data.shape)


# In[12]:


print(data.describe)


# In[13]:


data = data.sample(frac = 0.1, random_state = 1)
print(data.shape)


# In[14]:


data.hist(figsize=(20,20))
plt.show()


# In[15]:


fraud=data[data['Class']==1]
valid=data[data['Class']==0]

outlier_fraction=len(fraud)/float(len(valid))
print(outlier_fraction)

print('fraud cases: {}'.format(len(fraud)))
print('valid cases: {}'.format(len(valid)))


# In[16]:


corrmat=data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# In[17]:


columns=data.columns.tolist()

columns=[c for c in columns if c not in ["Class"]]

target = "Class"

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)


# In[25]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state = 1

classifiers ={
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state = state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors=20,
    contamination=outlier_fraction)
}


# In[26]:


n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #RESHAPE THE PREDICTION
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




