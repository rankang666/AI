#coding=utf-8
#运行与测试

import numpy as np

from sklearn import datasets

from ann import tools
from ann import pretron


class ANNet:
    def __init__(self):
        (data, target) = datasets.load_iris(return_X_y=True)
        self.samples = data[0:100]
        self.labels = target[0:100]
        self.net = pretron.Pretron(
            input_size=4,
            output_size=1,
            learning_rate=0.1
        )
    def train(self,count):
        self.count =count
        for n in range(count):
            for i in range(self.samples.shape[0]):
                self.net.calculate_values(np.array([self.samples[i]]).T)
                self.net.update_weights(np.array([self.labels[i]]).T)

    def predict(self,s):
        return self.net.calculate_values(s)

# 准备数据
net = ANNet()
net.train(100)

# 预测
print(net.samples)
print(net.predict(np.array([net.samples[10]]).T))

# for i in range(net.samples.shape[0]):
#     print(net.predict(np.array([net.samples[i]]).T))
#     print(np.array([net.labels[i]]).T)






# 数学测试:
# m  = np.array(
#     [
#         [1,2,3,4],
#         [2,3,4,5]
#     ]
# )
#
# v = np.array([[0.1,0.2,0.3,0.4]])
# print(m.shape)
# print(v.shape)
# print(tools.mv_sum(m,v.T))

# 准备数据
# (data, target) = datasets.load_iris(return_X_y=True)
# samples =data[0:100]
# labels = target[0:100]

# 跑通流程
# net = pretron.Pretron(
#     input_size=4,
#     output_size=1,
#     learning_rate=0.1
# )
#
# net.calculate_values(np.array([samples[0]]).T)
# net.update_weights(np.array([labels[0]]).T)





