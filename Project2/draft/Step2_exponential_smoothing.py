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
# 1. reading data 
# =============================================================================
train = pd.read_csv('data/P2train.csv', parse_dates=['Time'],index_col='Time',header=0)
test = pd.read_csv('data/P2test.csv', parse_dates=['Time'],index_col='Time',header=0)
test_index = pd.read_csv('data/P2test_index.csv', header=0)

print(train.shape)
train.head(10)


# =============================================================================
# 2. moving average under assumptions that observations follow a constant trend modal 
# =============================================================================

df=train['2017-02-01':'2017-02-15']
y='TrafficVolume'

fig, ax = plt.subplots(2,2,figsize=(22,10))
fig.suptitle("Moving Average", fontsize=14)

win = 2;
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[0][0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0][0].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[0][0].title.set_text('window size = {}'.format(win))
win = 3; 
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[0][1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0][1].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[0][1].title.set_text('window size = {}'.format(win))
win = 10;
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[1][0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1][0].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[1][0].title.set_text('window size = {}'.format(win))
win = 24;
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[1][1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1][1].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[1][1].title.set_text('window size = {}'.format(win))



df=train['2017-02']
y='TrafficVolume'

fig, ax = plt.subplots(1,3,figsize=(22,5))
win = 24;
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[0].title.set_text('window size = {}'.format(win))

win = 168; # 24*7
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[1].title.set_text('window size = {}'.format(win))

win = 720; # 24*30
ma = df[y].rolling(window=win, min_periods=0, center=False)
ax[2].scatter(x=range(0,df.shape[0]), y=df[y])
ax[2].scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
ax[2].title.set_text('window size = {}'.format(win))


# =============================================================================
# 3.a) Exponentially weighted moving average under assumptions that observations follow a constant trend modal 
# =============================================================================
df=train['2017-02-01':'2017-02-15']
y='TrafficVolume'

fig, ax = plt.subplots(1,3,figsize=(22,5))
fig.suptitle("Exponential moving average", fontsize=14)
al = 0.1;

ewma = df[y].ewm(alpha=al, min_periods=0)
ax[0].scatter(x=range(0,df.shape[0]), y=df[y])
ax[0].scatter(x=range(0,df.shape[0]), y=ewma.mean(), color='blue')
ax[0].title.set_text('alpha= {}'.format(al))

al = 0.3;
ewma = df[y].ewm(alpha=al, min_periods=0)
ax[1].scatter(x=range(0,df.shape[0]), y=df[y])
ax[1].scatter(x=range(0,df.shape[0]), y=ewma.mean(), color='blue')
ax[1].title.set_text('alpha= {}'.format(al))

al = 0.5;
ewma = df[y].ewm(alpha=al, min_periods=0)
ax[2].scatter(x=range(0,df.shape[0]), y=df[y])
ax[2].scatter(x=range(0,df.shape[0]), y=ewma.mean(), color='blue')
ax[2].title.set_text('alpha= {}'.format(al))

# =============================================================================
# 3.b) selecting the smoothing parameters to minimize error SSE = sum[Y(i)-L(i-1)]²  #L(n) = alpha* Y(n) + (1-alpha)*L(n-1) = sum over i=1 to n of [(alpha*(1-alpha)^(n-i))*Y(i)]
# L(n-1) = previous mean values 
# if alpha = 1  then L(n)=Y(n) ie the prediction is equal to the data 
    
# =============================================================================
df=train
alpha = np.linspace(0.01,1,num=100)
err = [];
sses = pd.DataFrame()
for al in alpha:
    ewma = df[y].ewm(alpha=al, min_periods=0)  
    pred = ewma.mean();
    diff = df[y] - pred.shift(1);  ## we compare Y(i) and L(i-1)
    sse=np.sqrt(diff ** 2)
    sses['sse'+str(al)]=sse
    err.append(sse.mean())
    
plt.plot(alpha, err)
optal = alpha[np.argmin(err)]
plt.axvline(x=optal, color='red')
plt.title('prediction error given alpha with optimal alpha = {}'.format(optal))

# =============================================================================
# 3.c) assumptions verifications
# =============================================================================
## seasonal testing
alpha=0.05 # 95 % confidence intervals 
lags =150
#24*7=168
#168*30=5040
data =train['TrafficVolume']['2017']
fig, ax = plt.subplots(1,2,figsize=(22,6))
fig.suptitle('modal assumption checking with confidence interval of {} % '.format((1-alpha)*100))
fig = sgt.plot_acf(data, ax=ax[0], lags=lags, alpha=alpha, unbiased=True)
fig = sgt.plot_pacf(data, ax=ax[1], lags=lags, alpha=alpha, method='ols')

## trend testing
df=train
y='TrafficVolume'
wins =[24,24*7,24*7*30]
wins_exp = {24 : 'approx. day',24*7 :'approx. month', 24*7*30 : 'approx. year'}
for win in wins :
    fig, ax = plt.subplots(1,1,figsize=(22,4))
    ma = df[y].rolling(window=win, min_periods=0, center=False)
    ax.scatter(x=range(0,df.shape[0]), y=df[y])
    ax.scatter(x=range(0,df.shape[0]), y=ma.mean(), color='blue')
    ax.title.set_text('moving average with window size = {} ie {}'.format(win,wins_exp[win]))
    
# =============================================================================
# 4. Seasonal Trend Corrected Smoothing  Y(t)= b0+ SN(t) + e(t)
# ============================================================================
# supposed form of observation Y : Y(t)= B0 + beta1*t +e(t)
# prediction/level L : L(n) = alpha *Y(n) + (1-alpha)*(L(n-1)+ B(n-1))
# Trend termn B(n) = beta*(L(n)-L(n-1)) + (1-beta)*beta(n-1)
# given a series, alpha and beta, return series of smoothed points
def double_exponential_smoothing(series, alpha, beta, L0, B0):
    result = []
    for n in range(0, len(series)):
        val = series[n]
        if n==0:
            level = alpha*val + (1-alpha)*(L0+B0); ##level(1)
            trend = beta*(level-L0) + (1-beta)*B0; ##trend(1)
            last_level = level;
        else:
            level = alpha*val + (1-alpha)*(last_level+trend) #trend = B(n-1)
            trend = beta*(level-last_level) + (1-beta)*trend
            last_level = level;
            
        result.append(level)
    return result

alpha = 0.4; #
beta = 0.1; #
series = df[y].values
holt = double_exponential_smoothing(series, alpha, beta,series[0], series[1]-series[0])
fig = plt.figure(figsize=(20,8))
plt.scatter(x=range(0,df.shape[0]), y=df[y])
plt.scatter(x=range(0,df.shape[0]), y=holt, color='blue')

# =============================================================================
# testing 
# =============================================================================

# plot the moving average/std with window size = period
df=train['TrafficVolume']['2017-01']

moving = df.rolling(window=7, min_periods=None, center=False)
#Plot rolling statistics:
fig = plt.figure(figsize=(20,6))
orig = plt.plot(df, color='blue',label='Original')
mean = plt.plot(moving.mean(), color='red', label='Moving Mean')
std = plt.plot(moving.std(), color='black', label = 'Moving Std')
plt.legend(loc='best')
plt.title('Moving Mean & Standard Deviation')
plt.show()
##


# =============================================================================
# seasonal decomposition why freq =52 ? 
# =============================================================================
df=train['2017']

freq = 52
df=train['TrafficVolume']['2017']
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df,freq=freq)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

df=df['2017-01':'2017-02']
fig, ax = plt.subplots(4,1,figsize=(20,12))
plt.subplot(411)
plt.plot(df, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


