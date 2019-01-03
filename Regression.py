import math, quandl
import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'CWDffsBBk5tGZ__Dzdiz'
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open',  'Adj. High',   'Adj. Low',  'Adj. Close',  'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
print(df.shape)
df.dropna(inplace=True)
print(df.shape)
# print(df.head())

X = np.array(df.drop(['label'], axis=1))
X = preprocessing.scale(X)
y = np.array(df['label'])

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# clf = LinearRegression()
clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)