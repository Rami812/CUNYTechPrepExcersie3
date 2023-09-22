#!/usr/bin/env python
# coding: utf-8

# # Our first machine learning model: Logistic Regression

# In[90]:


# Import our libraries 
#Pandas and numpy for data wrangling
import pandas as pd
import numpy as np

# Seaborn / matplotlib for visualization 
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import seaborn as sns

# Helper function to split our data
from sklearn.model_selection import train_test_split

# This is our Logit model
from sklearn.linear_model import LogisticRegression

# Helper fuctions to evaluate our model.
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score


get_ipython().run_line_magic('matplotlib', 'inline')


# # Import and inspect the Titanic dataset.
# * Load the titanic data set into a pandas dataframe.

# In[91]:


# Load the titanic data set into a pandas dataframe.
df = pd.read_csv('data/titanic.csv')
df.head()


# ## Data dictionary
# <img src='https://miro.medium.com/max/1260/1*rr3UGlpEv_PSMc1pyqa4Uw.png'>

# # Identify which columns have null values. 
# Inspect which varibles may be good / not good for using as features based on null values. 
# 

# In[92]:


# Identify which columns have null values. 
df.isnull().sum()/len(df)
#Over 50% of deck columns are nulls so we should drop that column


# # Check to see if our data has any duplicate rows.
# If so, remove the duplicates.

# In[93]:


# Check to see if our data has any duplicate rows.
print("Number of duplicates", df.duplicated().sum())
print("Size of df With duplicates",len(df))
df.drop_duplicates(inplace=True)
print("Size of df without duplicates",len(df))


# # Use sns.pariplot to visualize.
# * Set the hue='survived'.

# In[94]:


# Use sns.pariplot to visualize.
#df=df.astype('float64')
print(df.shape)
sns.pairplot(df,hue="survived");


# # Feature Engineering
# For your first model, only include use the `fare` and `sex` as features.
# * Convert the `sex` feature to a continuous value by using `pd.get_dummies()`.
# * Drop the `sex_female` column as it is the identical inverse of `sex_male`. 
#     * Hint, you can use `drop_first=True` in the `pd.get_dummies()` function to have this done automatically.
# * Create a `selected_features` variable that is a list of `fare` and `sex_male`.  
# * Define your X and y variables.
#     * `X` is your selected features
#     * `y` is your target features (survived). 
# * Split your data into training and testing groups by using `train_test_split()`
#     * __IMPORTANT: In `train_test_split` set `random_state=45`, so when you make another model, you can run it on the same random split of data.__

# In[95]:


# Convert the sex column into a continuous variable by using pd.get_dummies
df=pd.get_dummies(data=df,columns=["sex"],drop_first=True)
#selected_features=["fare","sex_male"]


# In[96]:


X=df[selected_features]
y=df["survived"]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=45,test_size=0.2)


# # Select our features 
#    * only include use the `fare` and `sex_male` as features for this model.

# In[97]:


# Select our features
selected_features = ["fare","sex_male"]

# Set X to be the features we are going to use.
X = df[selected_features]

# Set y to be our target variable. 
y = df["survived"]


# # Split our data into the testing and training groups. 

# In[98]:


# Split our data into testing and training.
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=45,test_size=0.2)

# Print the length and width of our testing data.
print(X_train.shape, X_test.shape)


# # Build and train your model
# * Initialize an empty Logistic Regression model. 
# * Fit your model with your training data. 
# * Predict the values of your testing data

# In[99]:


# Initalize our model
model=LogisticRegression()

# Train our model using our training data.
model.fit(X=X_train, y=y_train)


# # Evaluate your model
# 1. Make predictions of your test data and save them as `y_pred`. 
# 1. Calculate and print the accuracy, precision, recall, and f1 scores of your model.
#     * Hint, sklearn provides helper functions for this.
# 1. Plot the confusion matrix of your predicted results. 
#     * How many True Positives and True Negatives did your model get?

# In[100]:


# 1. Make predictions of your test data and save them as `y_pred`. 
y_pred=model.predict(X_test)
y_pred


# In[101]:


# 2. Calculate and print the accuracy, precision, recall, and f1 scores of your model.

# Calculate our accuracy
accuracy = accuracy_score(y_test,y_pred)

# Calculate our precision score
precision = precision_score(y_test,y_pred)

# Calculate our recall score
recall = recall_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred)

# Print each of our scores to inspect performance.
print("Accuracy Score: %f" % accuracy)
print("Precision Score: %f" % precision)
print("Recall Score: %f" % recall)
print('F1 Score %f' % f1)


# In[102]:


# 1. Plot a confusion matrix of your predicted results. 
import matplotlib.pyplot as plt 
fig = plt.figure(figsize=(8,8))


cm=confusion_matrix(y_test,y_pred)
cm=cm.round(2)
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
ax=sns.heatmap(cm, annot=True, cmap='Greens',fmt="g")
plt.xlabel('Predicted')
plt.ylabel('Actual');


# In[103]:


# How many True Positives and True Negatives did your model get?
print(tn,'True Negatives and',tp ,'True Positives')


# # Create another model, call this `model_2`.  This time also include the p_class and embarked features. 
# 1. Run `pd.get_dummies()` on pclass and embarked of your DataFrame.
# 1. Update your `selected_features` to include the new pclass, embarked, sibsp, and parch features.
# 1. Define your `X` and `y` variables.
# 1. Break your data into training and testing groups.
#     * __IMPORTANT, In `train_test_split` set `random_state=45` so we will be using the same data rows as our first model__.
# 1. Initialize a new model, call this one `model_2`
# 1. Fit / Train your new model
# 1. Make predictions of your test data and save them as `y_pred`. 
# 1. Calculate and print the accuracy, precision, recall, and f1 scores of your model.
# 1. Plot the confusion matrix of your predicted results. 
#     * How many True Positives and True Negatives did your model get?
#     
# Compare the results to your first model. Which model had a better accuracy, recall, precision, and f1 score.

# In[107]:


df = pd.read_csv('data/titanic.csv')
df=pd.get_dummies(df,columns=["pclass","embarked"])
df.columns


# In[108]:


#df = pd.read_csv('data/titanic.csv')

# Run pd.get_dummies on pclass and embarked of your DataFrame.
#df=pd.get_dummies(df,columns=["pclass","embarked"])

# Update your `selected_features` to include the new pclass and embarked features. 
selected_features=['pclass_1', 'pclass_2', 'pclass_3','embarked_C', 'embarked_Q', 'embarked_S','sibsp', 'parch']
# Define your X and y variables
X=df[selected_features]
y=df["survived"]
# Split our data into testing and training.
# !!! Remeber to use the same random state as you used before
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=45)

# Initalize our model_2
model_2 = LogisticRegression()

# Fit / Train our model using our training data.
model_2.fit(X=X_train,y=y_train)
# Make new predicitions using our testing data. 
y_pred=model_2.predict(X_test)
# Calculate our accuracy
accuracy_2 = accuracy_score(y_test,y_pred)

# Calculate our precision score
precision_2 = precision_score(y_test,y_pred)

# Calculate our recall score
recall_2 = recall_score(y_test,y_pred)

# Calculate your f1-score
f1_2 = f1_score(y_test,y_pred)

# Print each of our scores to inspect performance.
print("Accuracy Score: %f" % accuracy_2)
print("Precision Score: %f" % precision_2)
print("Recall Score: %f" % recall_2)
print('F1 Score %f' % f1_2)

# Plot your confusion matrix.
cm=confusion_matrix(y_test,y_pred)
cm=cm.round(2)
tn, fp, fn, tp=confusion_matrix(y_test,y_pred).ravel()
ax=sns.heatmap(cm, annot=True, cmap='Greens',fmt="g")
plt.xlabel('Predicted')
plt.ylabel('Actual');



# # EXTRA CREDIT 1. 
# * Use age as a feature. 
# * How will you fill the null values?
#     * Hint, use `df.age.fillna(???)`
# * Make a new feature that 'traveled_alone'.  The sibsp and parch contain the amout of people they are traveling with. Mark everyone that has no sibsp or parch as traveled alone set to 1 and everyone else set to 0. 
#     * Once you have this traveled_alone column, you dont need to use the the sibsp and parch cols in your model.

# In[ ]:


df = pd.read_csv('data/titanic.csv')

# Run pd.get_dummies on sex, pclass, and embarked of your DataFrame.


# Fill null age values with mean age.


# Create new traveled_alone feature


# Update your `selected_features` to include the new traveled alone and age


# Define your X and y variables


# Split our data into testing and training.
# Remeber to use the same random state as you used before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)


# Initalize our model
model_3 = LogisticRegression()

# Fit / Train our model using our training data.

# Make new predicitions using our testing data. 


# Calculate our accuracy
accuracy_3 = 

# Calculate our precision score
precision_3 = 

# Calculate our recall score
recall_3 = 

# Calculate your f1-score
f1_3 = 

# Print each of our scores to inspect performance.
print("Accuracy Score: %f" % accuracy_3)
print("Precision Score: %f" % precision_3)
print("Recall Score: %f" % recall_3)
print('F1 Score %f' % f1_3)

# Plot your confusion matrix.
fig = plt.figure(figsize=(8,8))
plt.xlabel('Predicted')
plt.ylabel('Actual');


# # EXTRA CREDIT 2:  
# 
# Use stats models to create a summary report.  Interpret the results. 

# In[ ]:





# In[ ]:




