from matplotlib import pyplot as plt
import numpy as np

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