#coding=utf-8

import numpy as np

from _01network.ann import tools


class Pretron:
    def __init__(self,input_size, output_size,learning_rate):
        # 构造权重空间
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-0.1,0.1,size=(self.output_size,self.input_size))
        self.bias = np.zeros((self.output_size,1))
        # 输出空间
        self.values = np.zeros((self.output_size,1))
        # 激活函数的定义
        self.activity_functor = lambda x: x
        self.derivative_functor = lambda x:1
        self.learning_rate = learning_rate

    # 计算数据
    def calculate_values(self, input_data):
        self.input_data = input_data
        # print("数据计算")
        # print(self.input_data)

        # 1.计算累加和
        tools.mv_sum(self.weights,input_data,self.values)

        # 2.添加偏置项
        self.values += self.bias
        # print(self.values)
        # 3.激活函数运算
        self.values = self.activity_functor(self.values)


    # 更新权重
    def update_weights(self,sample_label):
        # 学习率
        # 计算误差(y-t)*激活函数的导数
        self.delta = (self.values- sample_label)* self.derivative_functor(self.values)
        # print(self.delta)
        # print(self.input_data)
        self.w_grad = np.dot(self.delta,self.input_data.T)
        # print(self.w_grad)


        # 计算梯度
        # 更新梯度
        # print("计算梯度更新权重")
        self.weights += self.w_grad
        # print(self.weights)

