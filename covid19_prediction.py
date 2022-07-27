# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:29:56 2022

@author: Shah
"""


from covid19_module import EDA,ModelDevelopment,ModelEvaluation,PlotFigure
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
import datetime
import pickle
import os
#%% Constant
CSV_PATH_TRAIN = os.path.join(os.getcwd(),'dataset',
                              'cases_malaysia_train.csv')

CSV_PATH_TEST = os.path.join(os.getcwd(),'dataset',
                              'cases_malaysia_test.csv')

MMS_PATH = os.path.join(os.getcwd(), 'model', 'mms_train.pkl')

LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))

BEST_MODEL_PATH = os.path.join(os.getcwd(), 'model', 'best_model.h5')
#%%STEP 1) Data Loading
df = pd.read_csv(CSV_PATH_TRAIN,na_values=[' ','?'])
df_test = pd.read_csv(CSV_PATH_TEST,na_values=[' ','?'])    
#%%STEP 2)Data Inspection

#train
df.head(10)
df.tail(10)
df.info()
df.isna().sum()
df.describe().T

eda=EDA()

eda.plot_trend(df.cases_new,win_start=98,win_stop=650)
    
#train data need to do interpolate, consist of 12 NaNs for cases_new
#test

df_test.head(10)
df_test.tail(10)
df_test.info()
df_test.isna().sum()
df_test.describe().T

eda.plot_trend(df_test.cases_new,win_start=98,win_stop=650)

#test data need to interpolate, consist of 1 NaNs for cases_new
#%%STEP 3) Data Cleaning
#interpolate
#train dataset
df=df.interpolate(method='linear')

df['cases_new'] = np.round(df['cases_new']).astype(int)
df.isna().sum()

eda.plot_trend(df.cases_new,win_start=98,win_stop=650)

#test dataset

df_test=df_test.interpolate(method='linear')

df_test['cases_new'] = np.round(df_test['cases_new']).astype(int)
df.isna().sum()

eda.plot_trend(df_test.cases_new,win_start=98,win_stop=650)

#%%STEP 4) Features Selection
#%%STEP 5) Data Preprocessing

#TRAIN DATASET
X = df['cases_new']#only 1 feature

mms = MinMaxScaler()
X = mms.fit_transform(np.expand_dims(X,axis=-1))

with open(MMS_PATH, 'wb') as file:
    pickle.dump(mms, file)

win_size = 30
X_train = []
y_train = []

for i in range(win_size,len(X)): 
    X_train.append(X[i-win_size:i]) 
    y_train.append(X[i])

X_train = np.array(X_train)
y_train = np.array(y_train)


#%%Test dataset

dataset_cat = pd.concat((df['cases_new'],df_test['cases_new']))

#concat
length_days = win_size + len(df_test)
tot_input = dataset_cat[-length_days:]

Xtest = mms.transform(np.expand_dims(tot_input, axis=-1)) 

X_test = []
y_test = []

for i in range(win_size,len(Xtest)):
    X_test.append(Xtest[i-win_size:i])
    y_test.append(Xtest[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

#%% Model development

input_shape = np.shape(X_train)[1:]

md=ModelDevelopment()

model=md.simple_dl_model(input_shape,nb_node=64,dropout_rate=0.3,
                    activation='relu')

plot_model(model,show_shapes=True,show_layer_names=True)

#%%
model.compile(optimizer='adam',loss='mse',
              metrics=['mean_absolute_percentage_error','mse'])

#callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)

# ModelCheckpoint # To get the best model
mdc = ModelCheckpoint(BEST_MODEL_PATH,
                      monitor='val_mean_absolute_percentage_error',
                      save_best_only=True,
                      mode='min',
                      verbose=1)

#%% Model Training
hist = model.fit(X_train,y_train,
                 epochs=300,
                 callbacks=[tensorboard_callback,mdc],
                 validation_data=(X_test,y_test))
                 
#%% Model Evaluation

#plt_figure
predicted_stock_price=model.predict(X_test)
actual_price=mms.inverse_transform(y_test)
predicted_price=mms.inverse_transform(predicted_stock_price)

pf = PlotFigure()
plot_fig = pf.plot_fig(y_test,predicted_stock_price,predicted_price,
             actual_price)


print(hist.history.keys())
key=list(hist.history.keys())

#plot_hist_graph
me=ModelEvaluation()

plot_hist_loss=me.plot_hist_graph(hist,key,0,3)

plot_hist_mse=me.plot_hist_graph(hist,key,2,5)

plot_hist_mape=me.plot_hist_graph(hist,key,1,4)

#plot figure
y_pred=model.predict(X_test)
actual_cases = mms.inverse_transform(y_test)
predicted_cases = mms.inverse_transform(y_pred)

pf = PlotFigure()

plot_fig = pf.plot_fig(y_test,y_pred,actual_cases,predicted_cases)
#%% MAE,MSE and MAPE

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print('mae :', mae)
print('mse :', mse)
print('mape :', mape)