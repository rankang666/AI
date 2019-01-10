#coding=utf-8

import numpy as np
from sklearn import datasets

activity_functor = lambda x: 1.0/(1 + np.exp(-x))
derivative_functor = lambda x:x(1-x)

class Layer:
    def __init__(self, input_size, output_size, a_functor, d_functor,learning_rate):
        # 1.训练数据
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.1,0.1,(
            self.output_size,self.input_size
        ))
        self.bias = np.random.uniform(-0.1,0.1,(
            self.output_size,1
        ))
        # 激活函数与激活函数的导数
        self.a_functor = a_functor
        self.d_functor = d_functor
        #输出数据
        self.values = np.zeros((self.output_size,1))
        #上一层的delta
        # self.delta

    def forward(self,input_data):
        # 存储数据
        self.input_data = input_data
        # 计算加权求和
        s = np.dot(self.weights, self.input_data) + self.bias
        # 计算激活函数
        self.values = self.a_functor(s)

    def backward(self,delta):
        # 1.更新权重(学习率, 计算梯度)
        # 1.1 计算梯度
        self.w_grad = np.dot(delta, self.input_data.T)
        self.b_grad = delta
        # 1.2 计算更新值
        # 1.3 更新权重
        self.weights -= self.learning_rate * self.w_grad
        self.b_grad -= self.learning_rate * self.b_grad

        # 2.计算上一层的误差项
        tmp_v = np.dot(self.weights.T,delta)
        self.delta = self.d_functor(self.input_data)* tmp_v




class MLPNet:
    def __init(self):
        # 超参数 (网络结构)
        paramters = [4,8,6,3]
        self.layers=[]
        for i in range(len(paramters)-1):
            layer = Layer(
                input_size=paramters[i],
                output_size=paramters[i+1],
                a_functor=activity_functor,
                d_functor=derivative_functor,
                learning_rate=0.01
            )
            self.layers.append(layer)
        # 训练参数 (训练有关的)

    def train(self,samples,labels,times):
        # 循环次数
        for num in range(1):# times
            # 循环样本
            for i in range(samples.shape[0]):
                s = np.array([samples[i]]).T
                # 1. 向前计算输出值(循环神经层)
                for layer in self.layers:
                    layer.forward(s)
                    y = layer.values
                    print(f"数据: \n{s}")
                # 2. 根据输出计算整个的输出的误差项(每次训练，只计算一次)
                # 取出最终的神经网络的输出值(s就是我们最终的输出值)
                # 输出层的误差项
                delta = (y-np.array([labels[i]]).T) * self.layers[-1].d_functor(y)
                # 3.反向传递误差项，并顺便更新权重(循环神经层)
                for layer in self.layers[::-1]:
                    layer.backward(delta)
                    delta = layer.delta



    def predict(self,sample):
        # 向前计算输出(最后一层输出)
        # 判定输出
        pass


# 构造网络
net = MLPNet()

# 加载数据
output_size = 3
data, target = datasets.load_iris(return_X_y=True)
# 格式化标签
labels= np.zeros((len(target),output_size))
for row in range(len(target)):
    label = target[row]
    label[row][label] = 1

# 执行训练
net.train(data,label,1)

# 执行准确率测试






















