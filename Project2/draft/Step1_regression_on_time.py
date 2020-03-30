# =============================================================================
# # import the packages and libraries
# =============================================================================
import pandas as pd
pd.options.display.max_columns=60

import numpy as np
import sys

import statsmodels.tsa.statespace as sts

from matplotlib import style
style.use('ggplot')

import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.tsa.arima_process as sta
import statsmodels.graphics.tsaplots as sgt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf

from util_formula import *

# =============================================================================
# 1. reading data and summary statistics
# =============================================================================
train = pd.read_csv('data/P2train.csv', parse_dates=['Time'],index_col='Time',header=0)
test = pd.read_csv('data/P2test.csv', parse_dates=['Time'],index_col='Time',header=0)
test_index = pd.read_csv('data/P2test_index.csv', header=0)

print(train.shape)
print(train.columns)
print(train.dtypes)
train.head(10)


train.describe()
#Outliers spot : max rainfall 9m/h temperature 0 K 

#The highest recorded rainfall in a single year was 22,987 mm (904.9 in) in 1861. The 38-year average at Mawsynram, Meghalaya, India is 11,873 mm (467.4 in).
train[train.Temp<200]  #2014-01-31 to 2014-02-02  
# if <200 <=> <-75 °C then put mean(n-1,n+1)
train[train.TrafficVolume<10]
train[train.Rain1h>2000]  #2016-07-11 17:00:00  here around = 0
#if >2000 then put mean (n-1,n+1)

#=============================================================================
# Exploration of the data
# =============================================================================

# plot the time series data
fig = plt.figure(figsize=(20,6))
plt.plot(train.Temp)

fig = plt.figure(figsize=(20,6))
plt.plot(train['2016'].Temp)

#All
fig = plt.figure(figsize=(20,6))
plt.plot(train.TrafficVolume)
plt.title('TrafficVolume by time')

fig = plt.figure(figsize=(20,6))
plt.plot(train['2016':].TrafficVolume)
plt.title('TrafficVolume by time for 2 years')

fig = plt.figure(figsize=(20,6))
plt.plot(train['2016'].TrafficVolume)
plt.title('TrafficVolume by time for 1 year')

fig = plt.figure(figsize=(20,6))
plt.plot(train['2016-03'].TrafficVolume)
plt.title('TrafficVolume by time for 1 month')

fig = plt.figure(figsize=(20,6))
plt.plot(train['2016-03-07':'2016-03-14'].TrafficVolume)
plt.title('TrafficVolume by time for 1 week')

pd.Timestamp('2016-03-12').day_name() #Saturday and Sunday lowest traffic

#3 days
fig = plt.figure(figsize=(20,6))
plt.plot(train['2016-03-04':'2016-03-07'].TrafficVolume)
plt.title('TrafficVolume by time for 3 days')
#1 day
fig = plt.figure(figsize=(20,6))
plt.plot(train['2016-03-04'].TrafficVolume)
plt.title('TrafficVolume by time for 1 day')


##Daily trend : peak 7-17
#Weekly trend : saturday and Sunday lowest traffic

train[str(train.TrafficVolume.idxmax())[:10]]

# =============================================================================
# #Regression on Time only
# =============================================================================

#Feature engineering
train1=train[['TrafficVolume']]
train1['index_no']= [x for x in range(0,len(train1))]
train1['hour']=train1.index.hour
train1['day_name']=train1.index.day_name()
#dayofweek : s0 = monday and 6 = sunday
train1['dayofweek']=train1.index.dayofweek
train1['dayofmonth']=train1.index.day
train1['dayofyear']=train1.index.dayofyear

train1['weekofyear']=train1.index.weekofyear
train1['monthofyear']=train1.index.month

train1['year']=train1.index.year-2012


#model training
feature_set=['C(hour)','C(dayofweek)','C(monthofyear)','year']

model = modelFitting('TrafficVolume', feature_set, train1)
model.summary()


'''function needed for calculating interval of prediction
    fit = modal 
    exog = new dataframe'''
def transform_exog_to_model(fit, exog):
    transform=True
    self=fit

    # The following is lifted straight from statsmodels.base.model.Results.predict()
    if transform and hasattr(self.model, 'formula') and exog is not None:
        from patsy import dmatrix
        exog = dmatrix(self.model.data.orig_exog.design_info.builder,
                       exog)

    if exog is not None:
        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or
                               self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)  # needed in count model shape[1]

    # end lifted code
    return exog


##prediction of 500 new datetime
lasttime=pd.Timestamp('2017-12-22 16:00:00')

x_pred_index_no = range(40000,40500)
x_pred_time = [lasttime+i*pd.Timedelta('1:00:00') for i in range (1, len(x_pred_index_no)+1)]

newdf = pd.DataFrame(index=x_pred_time,columns=['index_no'], data= x_pred_index_no)

newdf['year']=newdf.index.year-2012
newdf['monthofyear']=newdf.index.month
newdf['dayofmonth']=newdf.index.day
newdf['dayofweek']=newdf.index.dayofweek
newdf['hour']=newdf.index.hour

y_pred = model.predict(newdf)
transformed_exog = transform_exog_to_model(model, newdf)
from statsmodels.sandbox.regression.predstd import wls_prediction_std
prstd, iv_l, iv_u = wls_prediction_std(model, transformed_exog, weights=[1])

train1_partial=train1['2017-12':]
fig, ax = plt.subplots(figsize=(24, 6))
ax.plot(train1_partial['index_no'], train1_partial['TrafficVolume'])
ax.scatter(train1_partial['index_no'], train1_partial['TrafficVolume'])
fig.suptitle('Prediction Intervals')
ax.grid(True)
ax.plot(list(x_pred_index_no), y_pred, '-', color='red', linewidth=2)
# interval for observations
ax.fill_between(x_pred_index_no, iv_l, iv_u, color='#888888', alpha=0.3)
ax.axis('tight')
plt.show()

##currrent model 

model.rsquared_adj #0.8342458234468448
model.aic #649414.3643822919

##full model
possible_feature=['index_no','hour','C(hour)','dayofweek','C(dayofweek)', 'dayofmonth','C(dayofmonth)','dayofyear','C(dayofyear)','C(weekofyear)','weekofyear','C(monthofyear)','monthofyear','year']
modelfull = modelFitting('TrafficVolume', possible_feature, train1)
modelfull.rsquared_adj  #0.8455436886541244
modelfull.aic #647034.5910585566

##foward  selection
#modelfwd=forward('TrafficVolume', possible_feature, train1, criterion="AIC", fullmodel = None) ##FWD in the following order
possible_feature=['C(hour)', 'C(dayofweek)', 'C(dayofyear)', 'C(weekofyear)', 'C(dayofmonth)', 'C(monthofyear)', 'index_no', 'year', 'weekofyear']
modelfwd=modelFitting('TrafficVolume', possible_feature, train1)

modelfwd.rsquared_adj #0.8455436886541244
modelfwd.aic #647034.5910585565



# =============================================================================
# #diagnostic
# =============================================================================

import scipy.stats as stats      #for normal distrib

def diagnosisplot(lm,Features):
    '''plotting Histogram of normalized residuals
       quantile-quantile plot of the residuals
       residuals against fitted value
       partial plots'''
    #1-1Histogram of normalized residuals
    res = lm.resid
    f1 = plt.figure(figsize=(8,6))
    f1 = plt.hist(lm.resid_pearson,bins=20)
    f1 = plt.ylabel('Count')
    f1 = plt.xlabel('Normalized residuals') 

    #1-2 check the normality of the residuals
    #quantile-quantile plot of the residuals
    fig2 = plt.figure(figsize=(10,10))
    fig = sm.qqplot(lm.resid, stats.distributions.norm, line='r') 
    
    #1-3 residuals against fitted value
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111)
    ax.scatter(lm.fittedvalues,lm.resid)
    ax.axhline(y=0, linewidth=2, color = 'g')
    ax.set(xlabel='fitted values',ylabel='residuals')
    
    #2 partial plots
    for i in range(0,len(Features)):
        fig1 = plt.figure(figsize=(20,10))
        fig1 = sm.graphics.plot_regress_exog(lm, Features[i],fig=fig1)        

#plots for simple modal 
sns.set(font_scale=1)    
diagnosisplot(model,model.params.index.to_list()[1:
    2]) ##delete the "2" if want to see all variables 

#plots for simple modal with sqrt 



    
    