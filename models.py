import keras
from keras.layers import Dense, Dropout, LSTM, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, ZeroPadding1D, SimpleRNN, Bidirectional
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from settings import MAX_SEQUENCE_LENGTH, MAX_MIC, character_to_index, max_mic_buffer

import json
import numpy as np
import datetime

def conv_model():
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(
        5, input_shape = (MAX_SEQUENCE_LENGTH, len(character_to_index) + 1)
    ))

    model.add(Conv1D(
        64,
        kernel_size = 5,
        strides = 1,
        activation = 'relu',
        #input_shape = (MAX_SEQUENCE_LENGTH, len(character_to_index) + 1)
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    #model.add(Dropout(0.5))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def one_layer_conv_model(embed_length = (len(character_to_index)+1)):
    model = keras.models.Sequential()
    model.add(ZeroPadding1D(
        5, input_shape = (MAX_SEQUENCE_LENGTH, embed_length)
    ))
    model.add(Conv1D(
        64,
        kernel_size = 5,
        strides = 1,
        activation = 'relu',
        #input_shape = (MAX_SEQUENCE_LENGTH, len(character_to_index) + 1)
    ))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def bilstm_model(embed_length = (len(character_to_index)+1)):
    model = keras.models.Sequential()
    model.add(Bidirectional(LSTM(
        64
    ),input_shape=(MAX_SEQUENCE_LENGTH, embed_length)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

class EnsembleModel:
    def __init__(self,models,predict_method,max_mic_buffer=0.1):
        self.models = models
        self.predict_method = predict_method
        self.max_mic_buffer = max_mic_buffer
        
    def combine_predictions(self,predictions):

        if self.predict_method is 'average':
            return np.mean(predictions)
        elif self.predict_method is 'classify_then_average':
            actual_predictions = []
            for prediction in predictions:
                if prediction < MAX_MIC - self.max_mic_buffer:
                    actual_predictions.append(prediction)
            if float(len(actual_predictions))/float(len(predictions))>=0.49:
                return np.mean(predictions)
            else:
                return MAX_MIC
        else:
            print('predict_method not recognized')
            return -100
        
    def predict(self,test_x):
        all_predictions = []
        combined_predictions = []
        for model in self.models:
            all_predictions.append(model.predict(test_x))
        for i in range(len(test_x)):
            combined_predictions.append(self.combine_predictions([all_predictions[k][i] for k in range(len(self.models))]))
        return combined_predictions
    
    def evaluate(self,test_x,test_y):
        predictions = self.predict(test_x)
        correctly_classified_error = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(test_y, predictions) if actual < MAX_MIC and predicted < MAX_MIC - self.max_mic_buffer])    
        all_error = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(test_y, predictions)])    
        all_active_error = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(test_y, predictions) if actual < MAX_MIC])    
        return correctly_classified_error,all_active_error, all_error
    
    def evaluate_as_classifier(self,test_x,test_y):
        true_positives=0
        true_negatives=0
        false_positives=0
        false_negatives=0
        all_predicted=self.predict(test_x)
        for i in range(len(test_y)):
            actual=test_y[i]
            predicted=all_predicted[i]
            if actual<MAX_MIC-0.0001:
                if predicted<MAX_MIC - self.max_mic_buffer:
                    true_positives+=1
                else:
                    false_negatives+=1
            else:
                if predicted<MAX_MIC - self.max_mic_buffer:
                    false_positives += 1
                else:
                    true_negatives += 1
        return true_positives,true_negatives,false_positives,false_negatives
