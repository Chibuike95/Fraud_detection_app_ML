#!/usr/bin/env python
# coding: utf-8

# IMPORT LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
sns.set(style = "whitegrid")
import pickle
# for plotting graph in jupyter cell
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# READ

# In[2]:


df = pd.read_csv('/Users/mac/Downloads/AIML Dataset.csv')
#df.describe()
df.head()
#df.isnull().sum() #---null check


# EDA AND DATA SCIENCE

# In[3]:


#all count of types with isFraud both 0 and 1

fraud_type_df = fraud_type.reset_index(name='count')

fraud_type_df


# In[4]:


# Actual Frauds (ie, isFraud = 1)
isFraud_positive = df[df["isFraud"] == 1].groupby("type").size()

isFraud_positive_df = isFraud_positive.reset_index(name = "count")

isFraud_positive_df


# In[5]:


# Visual
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(5, 3), facecolor='none')
isFraud_positive_df.plot(kind="bar", color='skyblue')

plt.title("Fraud Type", fontsize=16, fontweight='bold')
plt.ylabel("Fraud Rate", fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()


# In[6]:


df.columns


# In[7]:


df["isFraud"].value_counts()


# In[8]:


df["isFlaggedFraud"].value_counts()


# In[9]:


#check for null
df.isnull().sum().sum()


# In[10]:


df.shape


# In[11]:


#to chehck %age of isfraud to total data
round((df["isFraud"].value_counts()[1]/df.shape[0]) * 100,2)


# In[12]:


# Set the style using Seaborn
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(7, 5))  # Set the figure size
df["type"].value_counts().plot(kind="bar", color=sns.color_palette("Blues", n_colors=len(df["type"].unique())))

# Title and labels
plt.title("Transaction Types", fontsize=16, fontweight='bold')
plt.xlabel("Transaction Type", fontsize=14)
plt.ylabel("Count", fontsize=14)

# Adding grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Customize ticks
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()


# In[13]:


# Set the style using Seaborn
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(7, 5), facecolor='none')  # Set the figure size and transparent background
fraud_by_type.plot(kind="bar", color='skyblue')

# Title and labels
plt.title("Fraud Rate by Type", fontsize=16, fontweight='bold')
plt.ylabel("Fraud Rate", fontsize=14)

# Adding grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Customize ticks
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()


# In[14]:


df["amount"].describe().astype(int)


# In[15]:


sns.histplot(np.log1p(df["amount"]),bins = 100, kde = True, color = "skyblue")
plt.title("Transaction Amount Distribution (log scale)", fontsize=16, fontweight='bold')
plt.xlabel("log Amount + 1)")
plt.show()


# In[16]:


sns.boxplot(data = df[df["amount"] <50000], x = "isFraud", y = "amount", color = "skyblue")
plt.title("Amt vs isFraud(<50k)", fontsize=16, fontweight='bold')
plt.show()


# In[17]:


df["BalanceOrigDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

df["BalanceDiffDest"] = df["oldbalanceDest"] - df["newbalanceDest"]


# In[18]:


(df["BalanceOrigDiff"] < 0).sum()


# In[19]:


(df["BalanceDiffDest"] < 0).sum()


# In[ ]:


#determine the unique steps

#df["step"].unique()


# In[20]:


frauds_per_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()

plt.plot(frauds_per_step.index, frauds_per_step.values, label = "fraud_per_step")

plt.xlabel("Step (Time)")

plt.ylabel("Number of Frauds")

plt.title("Frauds over Time")

plt.grid(True)

plt.show()


# In[32]:


df.drop(columns = "step", inplace = True) # from the above, WE SEE THAT TIME IS NOT NECESSARY--- step is insignificant in building the model: so its dropped


# In[21]:


#to get the top senders

df["nameOrig"].value_counts().head(10)


# In[22]:


#to get the top receivers

df["nameDest"].value_counts().head(10)


# In[23]:


fraud_users = df[df["isFraud"] == 1]["nameOrig"].value_counts().head(10)

fraud_users


# In[24]:


fraud_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

fraud_types["type"].value_counts()


# In[25]:


sns.countplot(data = fraud_types, x = "type", hue = "isFraud")
plt.title("Fraud Distribution in Transfer & Cash_out")
plt.show()


# In[26]:


corr = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]].corr
corr()


# In[27]:


#Create the heatmap
sns.heatmap(
    corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    cbar_kws={"shrink": .8},  # Adjust the color bar size
)
# Add title and adjust font size
plt.title("Correlation Matrix", fontsize=16)

# Display the plot
plt.show()


# In[56]:


zero_after_transfer = df[
        (df["oldbalanceOrg"] > 0) &
        (df["newbalanceOrig"] == 0) &
        (df["type"].isin(["TRANSFER", "CASH_OUT"]))
]


# In[29]:


len(zero_after_transfer)


# In[30]:


df["isFraud"].value_counts()


# In[57]:


df.head()


# In[58]:


#drop irrelevant columns for feature eng...
df_model = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis = 1)


# In[35]:


df_model.head()


# PREPROCESSING & FEATURE ENGINEERING

# In[86]:


#divide into Categorical and Numeric data
categorical = ["type"]
numeric = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


# In[91]:


df_model.columns


# In[89]:


y = df_model["isFraud"]
x = df_model.drop("isFraud", axis =1)


# In[63]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify =y)


# In[90]:


preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop = "first"), categorical)
    ],
    remainder = "drop"
)


# MODELING & PREDICTION

# In[73]:


pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight = "balanced", max_iter = 1000))
])


# In[74]:


pipeline.fit(x_train,y_train)


# In[75]:


y_pred = pipeline.predict(x_test)

#y_pred


# TEST

# In[76]:


print(classification_report(y_test, y_pred))


# In[77]:


confusion_matrix(y_test, y_pred)


# In[78]:


# pipeline = pipeline.score(x_test, y_test)*100

# print(f"{pipeline}%")

# Evaluate but don't overwrite the pipeline
accuracy = pipeline.score(x_test, y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Save the trained model
import joblib
joblib.dump(pipeline, "sample_fraud_detection_pipeline.pkl")


# PKL For STREAMLIT

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


print(type(pipeline))
print(hasattr(pipeline, "predict"))


# In[85]:


df.columns


# In[92]:


x

