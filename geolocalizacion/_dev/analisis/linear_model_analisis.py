# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:07:05 2023

@author: HernanHGM

Perform linear_model for multiple variables
"""
# %% IMPORT LIBRARIES
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

# %% LOAD DATA
df = pd.read_csv("E:\\duraton\\geolocalizacion\\_data\\fly\\enriquecida_elevation_weather\\all_grouped_data.csv",
                               index_col=False, 
                               encoding="ISO-8859-1")
df = df[df.name=='Gato']
df = df[df.flying_situation =='flying']

# %%
cm = df.corr(numeric_only = True)
# %%
x = df[['tempC', 'windspeedKmph', 'sunHour',
        'pressure', 'precipMM', 'humidity']]
y = df['bird_altitude']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)


print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
x = sm.add_constant(x) # adding a constant
 
model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)


