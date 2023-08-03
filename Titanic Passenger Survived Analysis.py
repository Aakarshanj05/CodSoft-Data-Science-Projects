#!/usr/bin/env python
# coding: utf-8

# # EDA - TITANIC SURVIVAL PREDICTION 

# Using the Titanic Dataset I'll be building a model that will predict whether a passenger on the Titanic has survived or not.

# 
# ## Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing the Data

# In[2]:


train = pd.read_csv("titanic.csv")


# In[3]:


train.head(10)


# In[4]:


train.describe()


# In[5]:


train.isnull()


# In the above Tabel True value means that there is Null value present whereas False means there is no null values present.

# In[6]:


train.tail()


# Comparing the Cabing section it is clear that passenger id 1305,1307,1308,1309 is having NaN value which is equal to True. Whereas Passenger Id 1306 is having a Cabin value - C105, So its value is marked as False. Same is applicable for Rest of the columns too

# In[7]:


train.shape


# In[8]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='cividis')


# From this heatmap we can see that max NaN values is from Cabin column followed by Age and then Fare.

# In[9]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex', data=train)


# #### The number of Not Survived passenger is approx 1.8times the no of Survived passenger

# .

# In[11]:


num_male = len(train[train['Sex'] == 'male'])
num_female = len(train[train['Sex'] == 'female'])


# In[12]:


num_male


# In[13]:


num_female


# In[14]:


plt.bar(['Male', 'Female'], [num_male, num_female])
plt.ylim(0, 300)
plt.xlabel('Sex')
plt.ylabel('Number of Passengers')
plt.title('Number of Males vs. Number of Females on the Ship')
plt.show()


# #### No. of male is also almost 1.8 times more than the number of Female on the ship 
# 
# #### From the above two observation we can make a guess that all the female on the ship were saved while no male pessenger or very few male passenger were saved .

# .

# In[15]:


children_df = train[train['Age'] < 15]


# In[16]:


children_df


# In[17]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex', data=children_df)


# In[18]:


num_children = len(children_df)


# In[19]:


num_children


# In[20]:


num_male_children = len(children_df[children_df['Sex'] == 'male'])
num_female_children = len(children_df[children_df['Sex'] == 'female'])


# In[21]:


num_male_children 


# In[22]:


num_female_children


# #### This shows that passengers were only saved on the basis of Gender. It dosen't matter whether the children is below 1year or 14 years. 

# .

# In[23]:


female_mask = train['Sex'] == 'female'
survived_1_mask = train['Survived'] == 1


# In[24]:


result= train[female_mask & survived_1_mask]


# In[25]:


result.head(50)


# In[26]:


male_mask = train['Sex'] == 'male'
survived_mask_0 = train['Survived'] == 0


# In[27]:


result= train[male_mask & survived_mask_0]


# In[28]:


result.head()


# In[29]:


result = train[(train['Survived'] == 0) & (train['Sex'] == 'male')]
total_rows = result.shape[0]
print("Total number of rows where 'Survived' is 0 and 'S' is 1:", total_rows) 


# In[30]:


result = train[(train['Survived'] == 0) & (train['Sex'] == 'female')]
total_rows = result.shape[0]
print("Total number of rows where 'Survived' is 0 and 'S' is 0:", total_rows)


# In[31]:


result = train[(train['Survived'] == 1) & (train['Sex'] == 'female')]
total_rows = result.shape[0]
print("Total number of rows where 'Survived' is 1 and 'S' is 0:", total_rows)


# ## Based on the following calculation we can predict following things. 

# 1. In Titanic Dataset there were 266 Male and 152 Female .
# 
# 2. None of the Male passenger Survived whereas all the Female Passengers Survived. It suggests that females were given priority during the rescue efforts as per the dataset
# 

# In[32]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', hue= 'Pclass', data=train, palette='rainbow')


# In[33]:


sns.distplot(train['Age'].dropna(),bins=20)


# Maximum people are in the Age group of 20-30

# In[34]:


max_age=train['Age'].max()
min_age=train['Age'].min()


# In[35]:


max_age


# In[36]:


min_age


# Maximum Age of passenger was 76 and minimum age of passenger was 0.17

# In[37]:


sns.countplot(x='SibSp',data=train) 


# In[38]:


sns.countplot(x='Parch',data=train)


# In[39]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[40]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 43

        elif Pclass == 2:
            return 26

        else:
            return 24

    else:
        return Age


# In[41]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[42]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[43]:


train.drop('Cabin',axis=1,inplace=True)


# In[44]:


train.head()


# In[45]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[46]:


train.dropna(subset=['Fare'], inplace=True)


# In[47]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[48]:


train.head()


# ## Let's clean the Dataframe and remove unwated rows and do the Conversion of Categorical Features

# In[49]:


train.info()


# In[50]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[51]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[52]:


sex.head(10)


# In[53]:


train.head()


# In[54]:


embark.head()


# So, we have converted Male = 1 and Female = 0 . Now it will be easy for our Machine Learning Model to take it directly as Input 

# In[55]:


train.drop(['Sex', 'Embarked', 'Name', 'Ticket'],axis=1,inplace=True)


# In[56]:


train.head()


# In[57]:


train.head(10)


# In[58]:


train = pd.concat([train,sex,embark],axis=1)


# In[59]:


train.head()


# # Building a Logistic Regression model

# Lets split the data into training set and test set

# In[60]:


train.drop('Survived',axis=1).head()


# In[61]:


train['Survived'].head()


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.25, 
                                                    random_state=101)


# Here train.drop ('Survived', axis=1) will be used as input X for Machine Learning Model and train ['Survived'] will be used as target(y) for the machine learning model. 
# 
# Test size = 0.25 means that 25% data will be used for testing purpose or test set and remaining 75% will be used as training set

# # Training and Prediction

# ### Using Logistic Regression

# In[64]:


from sklearn.linear_model import LogisticRegression


# In[65]:


logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train,y_train)


# In[66]:


from sklearn.metrics import accuracy_score
predictions = logmodel.predict(X_test)
print('Accuracy:',accuracy_score(predictions,y_test))


# In[67]:


predictions


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


accuracy=confusion_matrix(y_test,predictions)


# In[70]:


accuracy


# In[71]:


from sklearn.metrics import accuracy_score


# In[72]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[73]:


predictions


# ### Using KNN Classifier

# In[74]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred2))


# ### Using Decision Tree

# In[75]:


from sklearn import tree

dt_model = tree.DecisionTreeClassifier()
dt_model.fit(X_train, y_train)


# In[76]:


from sklearn.metrics import accuracy_score

prediction_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, prediction_dt)


# In[77]:


accuracy_dt


# In[78]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,predictions)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(y_test,predictions))


# In[79]:


results = pd.DataFrame({
    'Model' : ['Logistic Regression','Confusion Matrix','KNN Classifier', 'Decision Tree'],
    'Score': [1.00, 1.00, 0.54, 1.00]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()


# #### So we can use any top three model for our predictions
# 
# #### Also we can say that from above all predictions that if the Gender of the passenger is Male then he won't survive and if the Gender of the passenger is Female then She we survive.
# #### So gender is one of the best thing in this dataset to reach to our conclusion

# In[80]:


train.head()


# In[81]:


train.count()


# In[82]:


train.head(25)


# In[83]:


train = train.rename(columns={'male': 'G'})


# In[84]:


train.head()


# In[85]:


result = train[(train['Survived'] == 1) & (train['G'] == 0)]

if not result.empty:
    print("There are rows where 'Survived' is 0 and 'S' is 1:")
    print(result)
else:
    print("No rows found where 'Survived' is 0 and 'S' is 1.")


# ## Conclusion

#  Through my analysis of the Titanic dataset, a clear trend emerged regarding the survival rates based on gender and age. It became evident that female passengers had a significantly higher chance of survival compared to their male counterparts. This observation suggests that, during the rescue process, priority was given to the safety of female passengers, indicating a possible gender-based advantage in survival.
# 
# Furthermore, when examining the fate of children, I noticed a striking disparity between female and male children. Female children were more likely to survive the disaster, while male children faced greater challenges in surviving.

# In[ ]:




