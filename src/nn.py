import keras
from keras.layers import Dense, Dropout, LSTM, Conv2D, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, ZeroPadding1D
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import json
from keras.models import model_from_json
import numpy as np
from src.settings import MAX_SEQUENCE_LENGTH, MAX_MIC, character_to_index, max_mic_buffer
from matplotlib import pyplot as plt
import datetime
from sklearn.metrics import mean_squared_error, r2_score
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

def draw_pic_result(log_num,predict,label_o):
    if log_num == 2:
        x_min = y_min = -2
        x_max = y_max = 14
        bias = 2
    elif log_num == 10:
        x_min = y_min = -1
        x_max = y_max = 4.5
        bias = 1
    else:
        x_min = y_min = 0
        x_max = y_max = 8196
        bias = 64
    x = np.arange(x_min-bias,x_max+bias,1)
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    plt.xlabel("Predict")
    plt.ylabel("Origin")
    plt.plot(x,x+0,'--',label="y=x")
    plt.plot(x,x+bias,'--',label="y=x+%d"%(bias))
    plt.plot(x,x-bias,'--',label="y=x-%d"%(bias))
    plt.scatter(predict,label_o,marker='o',color = 'blue', s=7)
    plt.legend(loc='upper left')
    plt.savefig("result_aureus.png")
    plt.clf()

def evaluate(model, test_x, test_y):
    predictions = model.predict(test_x)
    correctly_classified_error = np.mean([
        (actual - predicted) ** 2 
        for actual, predicted in zip(test_y, predictions)
        if actual < MAX_MIC and predicted < MAX_MIC - max_mic_buffer
    ])    
    all_error = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(test_y, predictions)])    
    all_active_error = np.mean([
        (actual - predicted) ** 2
        for actual, predicted in zip(test_y, predictions)
        if actual < MAX_MIC
    ])
    predict = []
    label_o = []
    for actual, predicted in zip(test_y, predictions):
        if actual < MAX_MIC and predicted < MAX_MIC - max_mic_buffer:
            predict.append(predicted)
            label_o.append(actual)
    print(r2_score(label_o,predict))
    draw_pic_result(10,predict,label_o)
    return correctly_classified_error, all_active_error, all_error
    
def evaluate_as_classifier(model, test_x, test_y, debug=False):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    all_predicted = model.predict(test_x)
    for i in range(len(test_y)):
        actual = test_y[i]
        predicted = all_predicted[i]
        if actual < MAX_MIC - 0.0001:
            if predicted < MAX_MIC - max_mic_buffer:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted < MAX_MIC - max_mic_buffer:
                false_positives += 1
                if debug == True:
                    print(vector_to_amp(test_x[i]))
                    print('predicted: ' + repr(predicted) + ', actual: '+repr(actual))
                    print('>p' + repr(false_positives) + '_' + repr(predicted))
                    print(vector_to_amp(test_x[i])['sequence'].replace('_', ''))
            else:
                true_negatives += 1
    return true_positives, true_negatives, false_positives, false_negatives


