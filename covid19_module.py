# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:32:15 2022

@author: Shah
"""

from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras import Input,Sequential 
import matplotlib.pyplot as plt

class EDA:
    def plot_trend(self,df,win_start=98,win_stop=650):
        df[win_start:win_stop]
        plt.plot(df)
        plt.title(df.name)
        plt.show()

class ModelDevelopment:
    def simple_dl_model(self,input_shape,nb_node=64,dropout_rate=0.3,
                        activation='relu'):
        '''
        

        Parameters
        ----------
        input_shape : TYPE
            DESCRIPTION.
        nb_node : TYPE, optional
            DESCRIPTION. The default is 64.
        dropout_rate : TYPE, optional
            DESCRIPTION. The default is 0.3.
        activation : TYPE, optional
            DESCRIPTION. The default is 'relu'.

        Returns
        -------
        model : TYPE
            DESCRIPTION.

        '''
        model = Sequential()
        model.add(Input(shape=(input_shape)))
        model.add(LSTM(64,return_sequences=(True)))
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        model.add(Dense(1,activation='relu'))
        model.summary()
        
        return model

class ModelEvaluation:
    def plot_hist_graph(self,hist,key,a=0,b=3):
        plt.figure()
        plt.plot(hist.history[key[a]])
        plt.plot(hist.history[key[b]])
        plt.legend(['training_'+ str(key[a]), key[b]])
        plt.show()
        
class PlotFigure:
  def plot_fig(self,y_test,y_pred,actual_cases,predicted_cases):
      plt.figure()
      plt.plot(y_test,color='red')
      plt.plot(y_pred,color='blue')
      plt.xlabel('Date')
      plt.ylabel('Covid19 Cases')
      plt.legend(['Actual','Predicted'])
      plt.show()
      
      plt.figure()
      plt.plot(actual_cases,color='red')
      plt.plot(predicted_cases,color='blue')
      plt.xlabel('Date')
      plt.ylabel('Covid19 Cases')
      plt.legend(['Actual','Predicted'])
      plt.show()