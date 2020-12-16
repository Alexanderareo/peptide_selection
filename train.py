from preprocess_cnn import get_bacterium_df,sequence_to_vector
from preprocess_cnn import add_random_negative_examples,value2log
from sklearn.model_selection import train_test_split
from evaluate import uniprot_precision
import numpy as np
import pandas as pd
from load_data import load_df_from_dbs
from models import conv_model
from evaluate import evaluate,evaluate_as_classifier

def train_model_cnn(bacterium, negatives_ratio=1, epochs=100, test_type="random"):
    """
    Bacterium can be E. coli, P. aeruginosa, etc.
    When with_negatives is False, classification error will be 0
    and error on correctly classified/active only/all will be equal
    because all peptides in the dataset are active
    """
    if test_type=="random":
        DATA_PATH = 'data_cnn/'
        df = load_df_from_dbs(DATA_PATH)
        bacterium_df = get_bacterium_df(bacterium, df) # 按菌种过滤后的条目
        print("Found %s sequences for %s" % (len(bacterium_df), bacterium))
        bacterium_df['vector'] = bacterium_df.sequence.apply(sequence_to_vector)  #新建一列存放映射之后的向量

        x = np.array(list(bacterium_df.vector.values))
        y = bacterium_df.value.values # label
        x, y = add_random_negative_examples(x, y, negatives_ratio) # 加入随机的负样例
        train_x, test_x, train_y, test_y = train_test_split(x, y,test_size=0.2)

    else:
        data = pd.read_csv(test_type,encoding="utf8")
        data["vector"] = data.sequence.apply(sequence_to_vector)
        data["value"] = data.MIC.apply(value2log)
        x = np.array(list(data.vector.values))
        y = data.value.values # label
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)


    model = conv_model()
    model.fit(train_x, train_y, epochs=epochs) # 模仿sklearn
    print("Avg. MIC error (correctly classified, active only, all)")
    print(evaluate(model, test_x, test_y))
    print('True positives, true negatives, false positives, false negatives')
    true_positives, true_negatives, false_positives, false_negatives = evaluate_as_classifier(
        model, test_x, test_y
    )
    print(true_positives, true_negatives, false_positives, false_negatives)
    print("Accuracy:",
        (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    )
    print("Precision on Uniprot:")
    print(uniprot_precision(model))

    return model

def train_model_lstm():
    return 0