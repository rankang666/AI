#coding=utf-8
import numpy as np
from sklearn import datasets
# import matplotlib.pyplot as plt

data, target = datasets.load_iris(return_X_y=True)
samples = data[0:150]
pre_labels = target[0:150]

# print(samples)
# print(labels)

# 超参数
i_size = 4
o_size=3
labels = np.zeros((len(pre_labels), o_size))
for n in range(len(pre_labels)):
    l = pre_labels[n] # 取值要么0,要么1
    labels[n][l] = 1
# print(labels)
learn_rate = 0.1

# 训练相关参数
a_functor = lambda x: 1.0/(1+np.exp(-x))
d_functor = lambda x: x*(1-x)

# 可视化激活函数
# x = np.linspace(-10,10,1000)
# # y = a_functor(x)
# y = d_functor(x)
# plt.plot(x,y,color='red')
# plt.show()





class OutLayer:
    def __init__(self, i_size, o_size,learn_rate):
        self.learn_rate = learn_rate
        # 训练相关局部参数
        self.input_size = i_size
        self.output_size = o_size
        # 条件数据
        self.w = np.random.uniform(
            -0.1,0.1,
            (self.output_size,self.input_size))
        self.b = np.zeros([self.output_size,1])
        # 中间计算数据
        # self.delta
        self.w_grad = np.zeros(self.w.shape, np.float32)
        self.b_grad = np.zeros(self.b.shape, np.float32)
        # 输出数据
        # self.v

    def forward(self,sample):
        self.input_data = sample
        # print("输入数据:\n{}".format(self.input_data))
        # 加权求和+偏置项
        a = np.dot(self.w, self.input_data.T)+ self.b
        # print(f"加权求和:\n{a}")

        # 激活函数运算
        self.v = a_functor(a)
        # print(f"计算结果: \n{self.v}")


    def backward(self,label):
        # print(f"标签: \n{label}")
        # 1. 计算误差项(y-t)y(1-y)
        self.delta = (self.v - label.T) * self.v * (1-self.v)
        # print(f"误差项: \n{self.delta}")
        # 2. 计算梯度 grad = delta * x b_grad = delta
        self.w_grad = np.dot(self.delta, self.input_data)
        self.b_grad = self.delta
        # print(f"梯度: \n{self.w_grad}")
        # 3. 更新权重 w -= aita * w_grad b -= aita*b.grad
        self.w -= self.learn_rate * self.w_grad
        self.b -= self.b_grad
        # print(f"新的权重: \n{self.w}")


layer = OutLayer(
    i_size = i_size,
    o_size = o_size,
    learn_rate= learn_rate)

amount = 1000
def train():
    # 循环调用样本反复训练(训练一个样本)
    for idx in range(amount):
        for num in range(len(pre_labels)):
            layer.forward(np.array([samples[num]]))
            layer.backward(np.array([labels[num]]))

def getMax_index(v):
    # 改变向量的维度
    v_ = v.reshape(v.shape[0]*v.shape[1])
    m = v_[0]
    idx = 0
    for i in range(v.shape[0]*v.shape[1]):
        if m < v_[i]:
            m = v_[i]
            idx = i
    return (m, idx)

def predict():
    # 计算输出
    counter = 0
    for num in range(len(pre_labels)): # len(pre_labels)
        layer.forward(np.array([samples[num]]))

        (m, idx) = getMax_index(layer.v)
        # if m > 0.5 and idx == pre_labels[num]:
        if idx == pre_labels[num]:
            counter += 1
    print(f"识别正确数量({counter}), 识别率（{100.0*counter/len(pre_labels)}%)")

    # 根据输出值，做预测规则


train()
predict()





