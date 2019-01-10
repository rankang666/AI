#coding=utf-8

import numpy as np
from sklearn import datasets
import tensorflow as tf

# 1. 准备数据
data, target = datasets.load_digits(return_X_y=True)
# print(data.shape)
# print(target)
# 训练方案
train_sample = data[0:1000]
test_sample = data[1000:]
# 标签数据格式化
lables = np.zeros((len(target),10),np.float32)
for i in range(len(target)):
    label = target[i]
    lables[i][label] = 1

train_labels = lables[0:1000]
test_labels = lables[1000:]

# 2. 描述神经网络
# 2.1 描述训练参数
# 2.1.1 训练参数
# (第一层)
# init_value= tf.random_normal((),stddev=0.01)
init_value = np.random.uniform(-0.1,0.1,(64,32))
w1 = tf.Variable(init_value, dtype=tf.float32)

init_value = np.zeros((1,32),dtype=np.float32)
b1 = tf.Variable(init_value,dtype=tf.float32)
# (第二层)
init_value = np.random.uniform(-0.1,0.1,(32,10))
w2 = tf.Variable(init_value, dtype=tf.float32)

init_value = np.zeros((1,10),dtype=np.float32)
b2 = tf.Variable(init_value,dtype=tf.float32)

# 2.1.2 输入与输出的站位参数
x = tf.placeholder("float",(None,64))
t = tf.placeholder("float", (None,10))

# 2.2 过程描述
# 2.2.1 加权求和描述
o_v1 = tf.sigmoid(tf.matmul(x, w1) + b1)
y= tf.sigmoid(tf.matmul(o_v1, w2) + b2)
# 2.2.2 损失描述
loss = 1.0/2*(y-t)*(y-t)
# 2.2.3 训练描述
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 3. 运行网络
session = tf.Session()
# 初始化全局变量(初始化训练参数)
session.run(tf.global_variables_initializer())
# tf.initialize_all_variables(session=session)

# (开始训练)
times = 100000
for num in range(times):
    session.run(train_op,feed_dict={x:train_sample, t:train_labels})
    # weight1 = session.run(w1)

# 预测
# 执行输出值计算，传递参数测试样本
result = session.run(y,feed_dict={x:test_sample})
# print(result)

max_idx_result = tf.argmax(result,axis=1)
max_idx_label = tf.argmax(test_labels,axis=1)
mr = session.run(max_idx_result)
ml = session.run(max_idx_label)
# print(mr)
# print(ml)
re = np.mean(mr == ml)
print(re)