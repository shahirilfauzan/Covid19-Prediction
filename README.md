# Covid19-Prediction
 To predict new cases of covid19 in Malaysia using the past 30 days of covid19 cases

## Project Description
The year 2020 was a catastrophic year for humanity. Pneumonia of unknown aetiology was first reported in December 2019., since then, COVID-19 spread to the whole world and became a global pandemic. More than 200 countries were affected due to pandemic and many countries were trying to save precious lives of their people by imposing travel restrictions, quarantines, social distances, event postponements and lockdowns to prevent the spread of the virus. However, due to lackadaisical attitude, efforts attempted by the governments were jeopardised, thus, predisposing to the wide spread of virus and lost of lives.

The scientists believed that the absence of AI assisted automated tracking and predicting system is the cause of the wide spread of COVID-19 pandemic. Hence, the scientist proposed the usage of deep learning model to predict the daily COVID cases to determine if travel bans should be imposed or rescinded

Thus, this project came to light by creating a LSTM Neural Network model to predict new cases in Malaysia using the past 30 days of number of cases

# How to install and run the project
1)Click on this provided link, you will be redirected to [GoogleColab](https://colab.research.google.com/drive/14LJyx3_FTVT9kbKioHFe_Un1-FHtm7Fd?usp=sharing),
you may need to sign in to your Google account to access it.

2)You also can run the model by using spyder. By downloading all the folder and files insde the repository, you can open the latest version of spyder and running the covid19_prediction.py to run the model. Ensure that covid19_module.py are in the same path as covid19_prediction.py.

For pc user: 
Software required: Spyder, Python(preferably the latest version) 
Modules needed: Tensorflow, Sklearn

# Model Architecture

![PlotModel](https://github.com/shahirilfauzan/Covid19-Prediction/blob/a0cb62fbe06c0406ae1f2f8f147418318a2a9955/static/model.png)

# Execution

## Training Loss

![PlotLoss](https://github.com/shahirilfauzan/Covid19-Prediction/blob/a0cb62fbe06c0406ae1f2f8f147418318a2a9955/static/loss_graph.png)

## Training Mean Absolute Percentage Error (MAPE)

![PlotLoss](https://github.com/shahirilfauzan/Covid19-Prediction/blob/a0cb62fbe06c0406ae1f2f8f147418318a2a9955/static/mape_graph.png)

# Tensorboard

![Tensorboard](https://github.com/shahirilfauzan/Covid19-Prediction/blob/a0cb62fbe06c0406ae1f2f8f147418318a2a9955/static/tensorboard_result.PNG)

# Project Outcome
As attached below the result of the model obtained Training Mean Absolute Percentage Error with 7.92%

![MAPE](https://github.com/shahirilfauzan/Covid19-Prediction/blob/a0cb62fbe06c0406ae1f2f8f147418318a2a9955/static/mape_result.PNG)

Actual vs Prediction Result :

![Result](https://github.com/shahirilfauzan/Covid19-Prediction/blob/a0cb62fbe06c0406ae1f2f8f147418318a2a9955/static/covid19_result_inverse.png)

# Powered by

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)

# Credit
Special thanks to [Ministry of Health Malaysia](https://github.com/MoH-Malaysia) by providing this [Dataset](https://github.com/MoH-Malaysia/covid19-public)
