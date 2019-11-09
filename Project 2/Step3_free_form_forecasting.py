# =============================================================================
# # import the packages and libraries
# =============================================================================
import pandas as pd
pd.options.display.max_columns=60
pd.options.display.max_rows = 999
import numpy as np
import sys

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.arima_process as sta
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.statespace as sts
from statsmodels.tsa.stattools import acf, pacf,kpss,adfuller

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns

from util_formula import *

# =============================================================================
# 1. reading data 
# =============================================================================
train = pd.read_csv('data/P2train.csv', parse_dates=['Time'],index_col='Time',header=0)
test = pd.read_csv('data/P2test.csv', parse_dates=['Time'],index_col='Time',header=0)
test_index = pd.read_csv('data/P2test_index.csv', header=0)

# =============================================================================
# Data Quality 
# =============================================================================

# =============================================================================
# 0-IsHoliday 
#seems to be only annotated for hour 0 of the related day
index= train[train.IsHoliday!='None'].index
for i in index:
    print(str(i)[0:10])
    train.loc[str(i)[0:10],'IsHoliday']='Holiday'
#some note even annotated ex : 2017-01-01 
train.loc['2017-01-01','IsHoliday']='Holiday'

#=============================================================================
#1- Rain1h
#2016-07-11 17:00:00 have rainhour >9000 while surrounding values of rain is 0 ==>error we will put 0 
train.loc['2016-07-11 17','Rain1h']=0

# =============================================================================
#2- Temp
#Temperature with 0 K => impossible. We'll replace 0 with nan and apply interpolation
train['Temp']=train['Temp'].replace(0, np.nan).interpolate()

# =============================================================================
#3- Time series 
train.loc['2017-02-28 17'] ##gives 3 values 

#add extra features 
train['date'] = [train.index[i].date() for i in range(0,len(train))]
train['year'] = [train.index[i].year for i in range(0,len(train))]
train['month'] = [train.index[i].month for i in range(0,len(train))]
train['day'] = [train.index[i].day for i in range(0,len(train))]
train['hour'] = [train.index[i].hour for i in range(0,len(train))]

#count by year / month 
train['year'].value_counts()
train['2017']['month'].value_counts()
(train['2017'].groupby(['month','day']).size()<24).sum()

#count number of values by day 
train[['date','TrafficVolume']].groupby(by='date').count().sort_values(by='TrafficVolume') #values ranges from 81 to 1
train.loc['2012-12-16'] ## 81 
train.loc['2012-12-16 10:00'] ##for a same hour, only the weathermain and weather description changes, Temp and Traffic vol remain the same 

##same for test set : 
test['date'] = [test.index[i].date() for i in range(0,len(test))]

#count number of values by day 
test[['date','TrafficVolume']].groupby(by='date').count().sort_values(by='TrafficVolume') #values ranges from 81 to 1

# =============================================================================
#3- Cleaning and resampling time 
##a-resample to have a clean data every hour 
##b-for multiple entries in an hour, took first values (impact on WeatherMain and Weather Description)
##c-fill the nan by interpolation
train_cleaned=train.resample('H').first().interpolate() 
train.resample('H').first().isnull().sum() #nber of nan filled 

# =============================================================================
# Linear regression with features 
# =============================================================================
import statsmodels.formula.api as smf
import util_formula

# =============================================================================
# feature engineering
# =============================================================================

train.drop(columns=['WeatherDescription'],inplace=True)
train['IsHoliday']= np.where(train['IsHoliday']=='None', 0,1)

y='TrafficVolume'
feature_set=['IsHoliday', 'Temp', 'Rain1h', 'Snow1h', 'CloudsAll', 'C(WeatherMain)']
             
model=modelFitting(y, feature_set, train)
model.summary()

forward=forward(y, feature_set, train)
##returns ['Temp', 'C(WeatherMain)', 'IsHoliday', 'CloudsAll', 'Rain1h']

forward_interaction=forward_interaction(y, feature_set, train)
#returns ['Temp:C(WeatherMain)', 'CloudsAll:C(WeatherMain)', 'IsHoliday', 'C(WeatherMain)', 'Rain1h:CloudsAll', 'Temp:CloudsAll', 'IsHoliday:Rain1h', 'Rain1h:Snow1h']
forward_interaction.summary()


##compute the variance not explained 
train['error']=train.TrafficVolume-forward_interaction.predict(train)
train['predict']=forward_interaction.predict(train)


def thousands(x):
    return int(x/1000)*1000
train.Traffictest=train.Traffictest.apply(thousands)

train.groupby(['WeatherMain','Traffictest']).size()


# =============================================================================
# Application of ARMA to this error
# =============================================================================

# =============================================================================
# stationarity test KPSS
# =============================================================================
'''Kwiatkowski-Phillips-Schmidt-Shin test for stationarity:
Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null hypothesis that x is level or trend stationary.'''

data=train_cleaned['2017'][['TrafficVolume']]

result = kpss(data.TrafficVolume.values, regression='c') # c H0=data is stationary around a constant (default).
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
#The p-value is greater than 0.05. The null hypothesis of stationarity around a cste is not rejected! 
    
# =============================================================================
# Augmented Dickey Fuller test (ADH Test)
# =============================================================================
'''Augmented Dickey-Fuller unit root test
The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.
The null hypothesis of the Augmented Dickey-Fuller is that there is a 
unit root.'''

result = adfuller(data.TrafficVolume.values, autolag='AIC',regression='c')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')
## null hypothesis rejected, meaning that the datasets is stationary ! 

# =============================================================================
# differencing if not stationary
# =============================================================================
alpha=0.05 ## 95 % for the confidence window
lags=80
y='TrafficVolume'
L=24   # period of L

## take the first regular difference
#data = train[y] - train[y].shift(1)
#data.dropna(inplace=True)
#fig, ax = plt.subplots(1,2,figsize=(20,10))
#fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
#fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')

#
## take the first seasonal difference
#data = train[y] - train[y].shift(L)
#data.dropna(inplace=True)
#fig, ax = plt.subplots(1,2,figsize=(20,10))
#fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
#fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')

## take the first regular and seasonal difference
#data = train[y] - train[y].shift(1) - (train[y].shift(L) - train[y].shift(L+1))
#data.dropna(inplace=True)
#fig, ax = plt.subplots(1,2,figsize=(20,10))
#fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
#fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')

# take the first regular difference cleaned
data = train_cleaned[y] - train_cleaned[y].shift(1)
data.dropna(inplace=True)
fig, ax = plt.subplots(1,2,figsize=(20,10))
fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')

# take the first seasonal difference cleaned
data = train_cleaned[y] - train_cleaned[y].shift(L)
data.dropna(inplace=True)
fig, ax = plt.subplots(1,2,figsize=(20,10))
fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')

# take the first regular and seasonal difference cleaned
data = train_cleaned[y] - train_cleaned[y].shift(1) - (train_cleaned[y].shift(L) - train_cleaned[y].shift(L+1))
data.dropna(inplace=True)
fig, ax = plt.subplots(1,2,figsize=(20,10))
fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')


###ARIMA 
from statsmodels.tsa.arima_model import ARIMA
# initial fit of the model without seasonal component

data=train_cleaned['2017-01'][['TrafficVolume']]
model = ARIMA(data, order=(1, 0, 1))  
results_ns = model.fit(disp=-1) 
print(results_ns.summary())

fig = plt.figure(figsize=(12,5))
plt.plot(data, label='Original',color='orange')
plt.plot(results_ns.fittedvalues, color='blue', label='fitted')
plt.legend(loc='best')
MAPE=sum(abs(results_ns.fittedvalues-data[y].values)/data[y].values)/data[y].shape[0]
plt.title('MAPE: %.4f'% MAPE)



## SARIMA
import statsmodels.api as sm
data=train_cleaned['2017'][['TrafficVolume']]
model = sm.tsa.SARIMAX(data, trend='n', order=(1,1,1), seasonal_order=(0,0,0,24))#,simple_differencing=True)
result_sm = model.fit()

print(result_sm.summary())
fitval = result_sm.fittedvalues

fig = plt.figure(figsize=(12,5))
plt.plot(data, label='Original')
plt.plot(fitval, color='blue', label='fitted')
plt.legend(loc='best')

### DIAGNOSTICS
import scipy.stats as stats
res = result_sm.resid
res = res[25:,]
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(res)
fig = sm.qqplot(res, stats.distributions.norm, line='r', ax=ax[1]) 


fig, ax = plt.subplots(1,2,figsize=(12,6))
fig = sgt.plot_acf(res, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(res, ax=ax[1], lags=lags, alpha=alpha, method='ols')



# plot the moving average/std with window size = period
df=train
y='TrafficVolume'
df=df[[y]]

df=df['2017-02']
fig, ax = plt.subplots(1,1,figsize=(22,6))
ax.scatter(x=range(0,df.shape[0]), y=df[y])
ax.axis('tight')

# take the differencing to make it more stationary
dif1 = df - df.shift(1)
dif2 = df - df.shift(2)
fig, ax = plt.subplots(1,2,figsize=(22,6))
ax[0].scatter(x=range(0,dif1.shape[0]), y=dif1[y])
ax[0].axis('tight')
ax[1].scatter(x=range(0,dif2.shape[0]), y=dif2[y])
ax[1].axis('tight')
plt.show()



# =============================================================================
#  Estimating ARMA parameters 
# =============================================================================
# =============================================================================
# ACF AND PACF
# =============================================================================

y='TrafficVolume'
df=train[[y]]  ##double bracket to have a dataframe type instead of series type

data = df['2017']
fig, ax = plt.subplots(2,1,figsize=(22,12))
fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')
# =============================================================================
# Nonseasonal modeling
# =============================================================================

