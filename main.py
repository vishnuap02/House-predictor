#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd

home_file_path = 'D:\machine learning\KarHouse.csv'
home_data = pd.read_csv(home_file_path)
var = home_data.columns
#Why i wasn't able to use name containing 'b' , which was showing 0x8


# In[60]:


# Import helpful libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
y = home_data.price

features = [ 'location', 'size', 'total_sqft', 'bath', 'balcony']

# Select columns corresponding to features, and preview the data
X = home_data.drop(['price','society'],axis=1)
print(X.head(20))
# Split into validation and training data
train_X,val_X , train_y, val_y = train_test_split(X, y,train_size=0.8,test_size=0.2, random_state=0)

#split columns based on whether they contain numeric values or string values 

catCols=[cname for cname in train_X.columns if train_X[cname].nunique()<10 and train_X[cname].dtype=='object']

numCols=[]
#[cname for cname in train_X.columns if train_X[cname].dtype in ['int64','float64']]

totCol=catCols+numCols

X_train=train_X[totCol].copy()
X_val=val_X[totCol].copy()


# In[37]:


# using Pipelines to preprocess and model , numerical and categorical colums separately
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numTrans=SimpleImputer(strategy="contant")

catTrans=Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numTrans, numCols),
        ('cat', catTrans, catCols)
    ])


# In[57]:


from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(n_estimators=100,random_state=0) 
    
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, train_y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_val)

# Evaluate the model
score = mean_absolute_error(val_y, preds)
print('MAE:', score)
print(home_data.head(1))


# In[58]:


data1= pd.DataFrame({'col1': [1, 2], 'col2': [0.1, 0.2]}, index=['a', 'b'])
print(type(data1))
#Need to design model to include location as well

data2=pd.DataFrame({'area_type':['Super built-up' ], 'availability':['19-Dec'], 'location':['Electronic City Phase II '], 'size':[2], 'society':['Electronic City Phase II '],
       'total_sqft':[1056], 'bath':[2], 'balcony':[1]})
prednow=my_pipeline.predict(data2)
print(prednow)


# In[65]:


import streamlit as st

st.write("#Bengaluru House Price Predictor Tool")

balcony = st.slider("Number of Balcony", 0, 10)
bath = st.slider("Number of bathrooms", 0, 10)
size=st.slider("Number of Bedrooms", 0, 10)
area_type=st.radio("Type of House Area",('Plot','Built-up','Super built-up'))
total_sqft=st.number_input("Built-up Area")
availability='Ready To Move'
society=''

#location= = st.text_input('Enter Location')
location='Electronic City Phase II'
#showing error and also in algo, im not taking area consideration

data2=pd.DataFrame({'area_type':[area_type], 'availability':[availability], 'location':[location], 'size':[size],
       'total_sqft':[total_sqft], 'bath':[bath], 'balcony':[balcony]})
predictPrice=my_pipeline.predict(data2)
print(predictPrice)
st.write("#The Entered values : ")
print(data2)

st.write("#The Expected Price of Home is Rs" )
st.write(predictPrice)


# In[ ]:


# size,total_sqft, bath, balcony=input('Enter Number of Bedrooms , built-up-area and number of balcony : ')
# area_type, availability, location, society=input('Enter Type of area , availability ,Location and Society Type : ')
# data2=pd.DataFrame({'area_type':[area_type], 'availability':[availability], 'location':[location], 'size':size],
# "society":[society], 'total_sqft':[total_sqft], 'bath':[bath], 'balcony':[balcony]}) Function for comparing
# different models def score_model(model, X_t=train_X, X_v=val_X, y_t=train_y, y_v=val_y): model.fit(X_t, y_t) preds
# = model.predict(X_v) return mean_absolute_error(y_v, preds)

#for i in range(0, len(models)):
 #   mae = score_model(models[i])
  #  print("Model %d MAE: %d" % (i+1, mae))

