import sys
import argparse
from train import train_model_cnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bacterium', type=str, default='E. coli', help='Name of bacterium, in single quotes')
    # 通过菌种筛选
    parser.add_argument('--negatives', type=float, default=0, help='Ratio of negatives to positives')
    # 调整比例
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--data',type=str, default='random') # 输入正样例数目即可
    args = parser.parse_args()
    train_model_cnn(bacterium=args.bacterium, negatives_ratio=args.negatives, epochs=args.epochs,test_type=args.data)