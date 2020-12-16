import numpy as np
from preprocess_cnn import load_uniprot_negatives,sequence_to_vector
from settings import MAX_SEQUENCE_LENGTH, MAX_MIC, character_to_index, max_mic_buffer

def uniprot_precision(model):
    """
    相当于真正的负样例测试集
    每次随机从uniprot数据集中选取负样例
    测试classifier的能力
    """
    negatives = load_uniprot_negatives(1000) # 每次调1000个负样例
    vectors = []
    for seq in negatives:
        try:
            vectors.append(sequence_to_vector(seq)) # 转化为矩阵
        except KeyError:
            continue
    preds = model.predict(np.array(vectors)) # 用模型预测
    # false_positives = len([p for p in preds if p < MAX_MIC - max_mic_buffer]) # 误差的数目
    false_positives = len([p for p in preds if p < 3.6])
    return 1 - false_positives / len(negatives)  # 负样例的预测正确率

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
    # predict = []
    # label_o = []
    # for actual, predicted in zip(test_y, predictions):
    #     if actual < MAX_MIC and predicted < MAX_MIC - max_mic_buffer:
    #         predict.append(predicted)
    #         label_o.append(actual)
    # print(r2_score(label_o,predict))
    # draw_pic_result(10,predict,label_o)
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
        if actual < MAX_MIC - 0.001:
            if predicted < MAX_MIC - max_mic_buffer:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if predicted < MAX_MIC - max_mic_buffer:
                false_positives += 1
                # if debug == True:
                #     print(vector_to_amp(test_x[i]))
                #     print('predicted: ' + repr(predicted) + ', actual: '+repr(actual))
                #     print('>p' + repr(false_positives) + '_' + repr(predicted))
                #     print(vector_to_amp(test_x[i])['sequence'].replace('_', ''))
            else:
                true_negatives += 1
    return true_positives, true_negatives, false_positives, false_negatives