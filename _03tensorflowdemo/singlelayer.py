#coding=utf-8

import numpy as np
from sklearn import datasets
import tensorflow as tf

# 1.描述训练参数
# 1.1 描述权重矩阵
init_value = np.random.uniform(-0.1,0.1,(4,2))
weights = tf.Variable(init_value,dtype=tf.float32)
# 1.2 描述偏置项
init_value = np.zeros((1,2))
bias = tf.Variable(init_value,dtype=tf.float32)
# 2.描述条件与结果
# 2.1 由于输入数据不需要预先分配空间，使用站位描述
x = tf.placeholder(tf.float32,(None, 4))
y = tf.placeholder(tf.float32,(None,2))
# 3.向前计算公式(加权求和 + 激活函数 )
values = tf.nn.sigmoid(tf.matmul(x,weights)+bias)
# 4.损失函数描述
loss = 1.0 / 2 * (y-values)*(y-values)
# 5.描述一个训练方法(梯度下降)
train_method = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

###########################################################################

# 6.准备环境

session = tf.InteractiveSession()
# 调用tensflow的默认全局初始化器
session.run(tf.global_variables_initializer())
# 7.运行(训练运行，变量操作运行)
# 训练次数times
times = 1000
# 准备数据
data, target = datasets.load_iris(return_X_y=True)
labels = np.zeros((len(target[0:100]),2))
for idx in range(len(target[0:100])):
    l = target[idx]
    labels[idx][l] = 1
for i in range(times):
    train_method.run(feed_dict={x:data[0:100],y:labels})

# 测试样本的输出

w_value = session.run(weights)
b_value = session.run(bias)
print(w_value, b_value)
result = session.run(values,feed_dict={
    x: data[0:100],
    weights: w_value,
    bias: b_value
})
print(result)

# 计算精确度
max_idx_op = tf.argmax(result,1)
max_idx = session.run(max_idx_op)
print(max_idx)
# acc = max_idx == target[100]
# print(acc)
# com_result = max_idx== labels
# mean = np.mean(com_result)
print(f"结果：{mean}")
################################################taskover