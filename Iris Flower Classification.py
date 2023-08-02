#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification 

# The Iris flower dataset consists of three species: Setosa, Versicolor,
# and Virginica. These species can be distinguished based on their
# measurements. 
# 
# Iris dataset to develop a model that can classify iris
# flowers into different species based on their sepal and petal
# measurements.

# ### Let's Import the required Libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Overview of the DataFrame

# In[2]:


df = pd.read_csv("Iris.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.columns


# In[6]:


df.head()


# In[7]:


df.tail()


# ### Let's Check the Null Values in the Dataset

# In[8]:


df.isna().sum()


# No Null Values Present in the Dataset

# In[9]:


df.info()


# In[10]:


df.shape


# In[11]:


df.sample(5)


# In[12]:


col1 = 'SepalLengthCm'
df[col1].hist()
plt.suptitle(col1)
plt.show()


# In[13]:


col2 = 'SepalWidthCm'
df[col2].hist()
plt.suptitle(col2)
plt.show()


# In[14]:


col3 = 'PetalLengthCm'
df[col3].hist()
plt.suptitle(col3)
plt.show()


# In[15]:


col4 = 'PetalWidthCm'
df[col4].hist()
plt.suptitle(col4)
plt.show()


# ## Relationship Between Column and Species

# In[16]:


col1 = 'SepalLengthCm'
sns.relplot(x=col1, y='Species', hue = 'Species', data = df)
plt.suptitle(col1, y=1.05)
plt.show()


# In[17]:


col2 = 'SepalWidthCm'
sns.relplot(x=col2, y='Species', hue = 'Species', data = df)
plt.suptitle(col2, y=1.05)
plt.show()


# In[18]:


col3 = 'PetalLengthCm'
sns.relplot(x=col3, y='Species', hue = 'Species', data = df)
plt.suptitle(col3, y=1.05)
plt.show()


# In[19]:


col3 = 'PetalWidthCm'
sns.relplot(x=col3, y='Species', hue = 'Species', data = df)
plt.suptitle(col3, y=1.05)
plt.show()


# From here It is clear that we can predict the Species of Setosa based on Petal Width or Petal Lenght as it do not overlap with other two species

# In[20]:


sns.pairplot(df, hue='Species')


# ## Train Test Split

# In[21]:


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()


# In[22]:


df['Species'] = le.fit_transform(df['Species'])
df.head(90)


# In[23]:


df['Species'].unique()


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


df_train, df_test =train_test_split(df, test_size = 0.25)


# In[26]:


df_train.shape


# In[27]:


df_test.shape


# In[28]:


df_test.head(10)


# In[29]:


df_train.head()


# ## Prepare Data For Modelling

# In[30]:


X_train=df_train.drop(columns=["Species","Id"]).values


# In[31]:


X_train.shape


# In[32]:


y_train=df_train["Species"].values
y_train


# In[33]:


train, test = train_test_split(df, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[34]:


X_test=df_test.drop(columns=['Species','Id']).values
y_test=df_test['Species'].values


# In[35]:


X_test.shape


# ## Manual Modelling based on pairplot

# In[36]:


df['Species']


# In[37]:


def single_feature_prediction(petal_length):
    if petal_length < 2.7: 
        return 0
    elif  petal_length < 4.9:
        return 1
    else:
        return 2


# In[38]:


df_train.columns


# In[39]:


X_train[:,3]


# In[40]:


manual_y_prediction =[single_feature_prediction(val) for val in X_train[:,3]]


# In[41]:


y_train


# In[42]:


manual_y_prediction == y_train


# In[43]:


manual_model_accuracy = np.mean(manual_y_prediction == y_train)


# In[44]:


manual_model_accuracy


# #### This is the manual Model Accuracy result

# .

# # Modelling

# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[46]:


model = LogisticRegression(max_iter=1000)


# In[47]:


model.fit(X_train,y_train)


# In[48]:


prediction = model.predict(X_test)
print('Accuracy:',accuracy_score(prediction,y_test))


# ### Using Confusion Matrix

# In[49]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(y_test,prediction))


# ### Using KNN Neighbors

# In[50]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred2))


# ### Using Decision Tree

# In[51]:


from sklearn import tree

dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[52]:


from sklearn.metrics import accuracy_score

prediction_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, prediction_dt)


# In[53]:


accuracy_dt


# In[64]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','KNN','Decision Tree' ],
    'Score': [0.97,0.97,0.97]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# All three are giving 97% results.

# In[55]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_model,X_train, y_train, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores


# In[56]:


y_test


# In[57]:


prediction_dt


# ### Creating Category

# Let's first enter the data manually

# In[58]:


category=['Setosa','Versicolor','Virginica']


# In[59]:


data = 5.7,3,4.2,1.1


# In[60]:


data_array = np.array([data])
data_array


# In[61]:


predic = dt_model.predict(data_array)


# In[62]:


print(category[int(predic[0])])


# ### Let's take the input from the user

# In[65]:


sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

# Convert the user input into a NumPy array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Use the trained model to predict the species of the flower
predicted_species = dt_model.predict(input_data)

# Display the predicted species to the user
print("Predicted Species:", predicted_species[0])


# #### It is clear that the Species is Setosa. 
# #### As 0 indicates to Setosa
# #### 1 indicates to Versicolor
# #### 2 indicates Virginica

# ## Conclusion
# 
# So, we were given different features of the flowers and have to make a model to predict the species of the flower based on random values of each features. 
# 
# The features (attributes) of the flowers are as follows:
# 
# Sepal Length (in centimeters)
# Sepal Width (in centimeters)
# Petal Length (in centimeters)
# Petal Width (in centimeters)
# 
# Throughout our analysis, we observed that the different species of Iris flowers exhibit distinct characteristics in terms of sepal and petal measurements. By using machine learning algorithms such as logistic regression, decision trees, KNN classification, we were able to build models that accurately classify Iris flowers into their respective species based on these measurements. 

# In[ ]:




