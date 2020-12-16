import pandas as pd
import numpy as np

data = pd.read_csv("positive_samples_4582.csv",encoding="utf8")
print(4582*0.75,4582*0.9)

data[:3437].to_csv("4582_train.csv")
data[3437:4124].to_csv("4582_val.csv")
data[4124:].to_csv("4582_test.csv")
