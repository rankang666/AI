#coding=utf-8
import tensorflow as tf
import os
import numpy as np
SIZE = 64

# 输入、输出数据的站位描述
x_data = tf.placeholder(tf.float32,[None,SIZE,SIZE,3]) # (个数， 高度，宽度， 深度)
y_data = tf.placeholder(tf.float32,[None,None])

# 把数据丢弃的概率做成临时输入数据
keep_prob_50 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

# 数据描述的模板(随机正态分布)
def weightVariable(shape):
    init = tf.random_normal(shape,stddev=0.01)
    return tf.Variable(init)
# 模板采用的是默认值随机正态分布
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

# 操作描述的基本模板
# 步长全部设置为1,补边原图与目标一样
def conv2d(x,w):
    return  tf.nn.conv2d(
        input=x, # (个数， 高度，宽度， 深度)
        filter=w, # (高度, 深度, 个数)
        # 卷积(x,y,..)
        strides=(1,1,1,1), # (前面卷积和在时隔维度上的步长)
        # 补边(和原图一样)
        padding="SAME")
# 选择最大池化
def maxPool(x):
    return tf.nn.max_pool(
        value=x, # (个数=1,高度, 宽度, 深度)
        # 池化卷积和的大小(图像对应)
        ksize=(1,2,2,1), # (个数, 高度, 宽度, 深度[上一层的卷积和的个数])
        # 池化图像减半
        strides=(1,2,2,1), # (个数维度步长, 高度维度的步长, 宽度维度的步长, 深度维度的步长)
        # 补边(和原图一样)
        padding="SAME") # 官方就两个VALID(不补边)和SAME(补边)
# 设置概率的门槛，决定样本是否参与训练
def dropOut(x,keep_prop):
    return tf.nn.dropout(x,keep_prop)

# 动态决定分类的数量
def CnnNet(classNum):
    # 网络的每一层训练参数数据描述 + 过程描述
    #============第一层(卷积层: 64*64*3, kernel: 3*3*3, 32)================
    # 1.卷积描述
    w1 = weightVariable((3,3,3,32)) # 卷积核的定义(高,宽,深,个数)
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x_data,w1) + b1) # 卷积运算
    # 2. 池化描述
    pool1 = maxPool(conv1)
    # 3. Dropout描述(性能)
    drop1 = dropOut(pool1, keep_prob_50)
    #============第二层(卷积层: 32*32*32, kernel: 3*3*32, 64)================
    w2 = weightVariable((3,3,32,64)) # 卷积核 (高,宽,深,个数)
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1,w2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropOut(pool2,keep_prob_50)

    #============第三层(卷积层: 16*16*64, kernel: 3*3*64, 64)================
    w3 = weightVariable((3, 3, 64, 64))  # 卷积核 (高,宽,深,个数)
    b3= biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, w3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropOut(pool3, keep_prob_50)

    # ============第四层(全连接)================
    w4 = weightVariable((8*8*64,512))
    b4 = biasVariable([512])
    # 做一个数据维度变化
    drop3_m = tf.reshape(drop3,(-1,8*8*64)) # -1与样本一样未知的数据样本数
    fc4 = tf.nn.relu(tf.matmul(drop3_m, w4) + b4)
    drop4 = dropOut(fc4,keep_prob_75)

    # ============第五层(输出)================
    w5 = weightVariable((512,classNum))
    b5 = biasVariable([classNum])
    out5 = tf.nn.relu(tf.matmul(drop4,w5) + b5)
    # 数据不用dropOut
    return out5 # 最终的输出
    # return out5, x_data, y_data, keep_prob_50, keep_prob_75  # 最终的输出


def train(xdata,ydata,savepath):
    '''
    :param x_data: 训练样本
    :param y_data:  样本标签: 期望的结果
    :param savepath: 保存训练的AI模型
    :return: 无
    '''
    # 构造一个网络
    out_net = CnnNet(2)
    # 执行网络
    # 定义优化器
    # 1. 损失函数(交叉熵损失计算函数, 不再使用均方差) 1/n * sum(y-t)^2
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=out_net,
            labels=y_data))
    # 2. 优化器(梯度下降)
    optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    # 训练结果保存期
    saver = tf.train.Saver()

    session = tf.Session()
    # 初始化全局变量(不包含站位变量)
    session.run(tf.global_variables_initializer())
    # 训练的次数
    times = 10
    batch_size = 10
    num_batch = len(xdata) // batch_size
    for n in range(times):
        # result = session.run(优化器,feed_dict={传递参数})
        # result = session.run(optimizer, feed_dict={x_data: xdata, y_data: ydata})
        for n_b in range(num_batch):
            batch_x = xdata[n_b*batch_size: (n_b+1) * batch_size]
            batch_y = ydata[n_b * batch_size: (n_b + 1) * batch_size]
            _, r_loss = session.run([optimizer, loss],
                                    feed_dict={
                                       x_data: batch_x,
                                       y_data: batch_y,
                                        keep_prob_50: 0.5,
                                        keep_prob_75: 0.75})
            print(f"损失值: {r_loss}")

    saver.save(session,savepath)


# 预测
def predict(session, data, predict):
    # predict = CnnNet(2)
    # 开始执行, 取得指定样本下的输出数据
    result = session.run([predict, tf.argmax(predict, 1)],
                         feed_dict={x_data: data, keep_prob_50: 1.0, keep_prob_75: 1.0})
    print(f"result: {result}")
    # 开始根据结果,匹配目录,并返回目录名(识别的用户名)

    # 语音设备
    return result


# 获取标签
def getfileandlabel(filedir):
    ''' get path and host paire and class index to name'''
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])
    # for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])

    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))

    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))
def onehot(numlist):
    ''' get one hot return host matrix is len * max+1 demensions'''
    b = np.zeros([len(numlist), max(numlist) + 1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()
# print(onehot([1,3,9]))
# print(getfileandlabel("../imgs"))

def getfilesinpath(filedir):
    ''' get all file from file directory'''
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getfilesinpath(os.path.join(path, diritem))