#coding=utf-8
import tensorflow as tf
import os
import numpy as np

# 图像大小
SIZE = 64




# 预测
def predict(session, test_x, predict):

    # 开始执行, 取得指定样本下的输出数据
    result = session.run([predict, tf.argmax(predict, 1)],
                         feed_dict={x_data: test_x, keep_prob_50: 1.0, keep_prob_75: 1.0})
    # num_pred = session.run(tf.argmax(result, 1))
    print("预测值:%d\n" % result[1][0])
    # print(f"result: {result}")
    # 开始根据结果,匹配目录,并返回目录名(识别的用户名)
    # text= "hello"
    # readText(text)

    # 语音设备
    return result




# 输入、输出数据的站位描述
x_data = tf.placeholder(tf.float32,[None,SIZE,SIZE,3]) # (个数， 高度，宽度， 深度)
y_data = tf.placeholder(tf.float32,[None,None])

# 把数据丢弃的概率做成临时输入数据
keep_prob_50 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)



class CnnNet:
    def __init__(self, train_x, train_y, savePath, classNum=None):
        self.train_x = train_x
        self.train_y = train_y
        self.savepath = savePath
        self.classNum = train_y.shape[1]
        if classNum != None:
            self.classNum = classNum

    def getCnnNet(classNum):

        def weightVariable(shape):
            ''' 数据描述的模板(随机正态分布) '''
            init = tf.random_normal(shape, stddev=0.01)
            return tf.Variable(init)
        def biasVariable(shape):
            ''' 模板采用的是默认值随机正态分布 '''
            init = tf.random_normal(shape)
            return tf.Variable(init)
        def conv2d(x, w):
            ''' 操作描述的基本模板: 步长全部设置为1,补边原图与目标一样'''
            return tf.nn.conv2d(
                input=x,  # (个数， 高度，宽度， 深度)
                filter=w,  # (高度, 深度, 个数)
                # 卷积(x,y,..)
                strides=(1, 1, 1, 1),  # (前面卷积和在时隔维度上的步长)
                # 补边(和原图一样)
                padding="SAME")
        def maxPool(x):
            ''' 选择最大池化 '''
            return tf.nn.max_pool(
                value=x,  # (个数=1,高度, 宽度, 深度)
                # 池化卷积和的大小(图像对应)
                ksize=(1, 2, 2, 1),  # (个数, 高度, 宽度, 深度[上一层的卷积和的个数])
                # 池化图像减半
                strides=(1, 2, 2, 1),  # (个数维度步长, 高度维度的步长, 宽度维度的步长, 深度维度的步长)
                # 补边(和原图一样)
                padding="SAME")  # 官方就两个VALID(不补边)和SAME(补边)
        def dropOut(x, keep_prop):
            ''' 设置概率的门槛，决定样本是否参与训练 '''
            return tf.nn.dropout(x, keep_prop)

        ''' 动态决定分类的数量 '''
        # 网络的每一层训练参数数据描述 + 过程描述
        # ============第一层(卷积层: 64*64*3, kernel: 3*3*3, 32)================
        # 1.卷积描述
        w1 = weightVariable((3, 3, 3, 32))  # 卷积核的定义(高,宽,深,个数)
        b1 = biasVariable([32])
        conv1 = tf.nn.relu(conv2d(x_data, w1) + b1)  # 卷积运算
        # 2. 池化描述
        pool1 = maxPool(conv1)
        # 3. Dropout描述(性能)
        drop1 = dropOut(pool1, keep_prob_50)
        # ============第二层(卷积层: 32*32*32, kernel: 3*3*32, 64)================
        w2 = weightVariable((3, 3, 32, 64))  # 卷积核 (高,宽,深,个数)
        b2 = biasVariable([64])
        conv2 = tf.nn.relu(conv2d(drop1, w2) + b2)
        pool2 = maxPool(conv2)
        drop2 = dropOut(pool2, keep_prob_50)

        # ============第三层(卷积层: 16*16*64, kernel: 3*3*64, 64)================
        w3 = weightVariable((3, 3, 64, 64))  # 卷积核 (高,宽,深,个数)
        b3 = biasVariable([64])
        conv3 = tf.nn.relu(conv2d(drop2, w3) + b3)
        pool3 = maxPool(conv3)
        drop3 = dropOut(pool3, keep_prob_50)

        # ============第四层(全连接)================
        w4 = weightVariable((8 * 8 * 64, 512))
        # w4 = weightVariable([8 * 16 * 32, 512])
        b4 = biasVariable([512])
        # 做一个数据维度变化
        drop3_flat = tf.reshape(drop3, (-1, 8 * 8 * 64))  # -1与样本一样未知的数据样本数
        # drop3_flat = tf.reshape(drop3, [-1, 8 * 16 * 32])
        fc4 = tf.nn.relu(tf.matmul(drop3_flat, w4) + b4)
        drop4 = dropOut(fc4, keep_prob_75)

        # ============第五层(输出)================
        w5 = weightVariable((512, classNum))
        b5 = biasVariable([classNum])
        out5 = tf.nn.relu(tf.matmul(drop4, w5) + b5)
        # out5 = tf.add(tf.matmul(drop4, w5), b5)
        # 数据不用dropOut
        print(out5)
        return out5  # 最终的输出
        # return out5, x_data, y_data, keep_prob_50, keep_prob_75  # 最终的输出

        # 训练

    def train(self, times = 10, xdata=None, ydata=None, savepath=None):
        '''
        :param x_data: 训练样本
        :param y_data:  样本标签: 期望的结果
        :param savepath: 保存训练的AI模型
        :return: 无
        '''
        if xdata != None:
            self.train_x = xdata
        if ydata != None:
            self.train_y = ydata
        if savepath != None:
            self.savepath = savepath

        # 构造一个网络
        out_net = CnnNet.getCnnNet(self.train_y.shape[1])
        print(f"创建{self.train_y.shape[1]}网络")
        # 执行网络
        # 定义优化器
        # 1. 损失函数(交叉熵损失计算函数, 不再使用均方差) 1/n * sum(y-t)^2
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=out_net,
                labels=y_data))
        # 2. 优化器(梯度下降)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        # 3. 精准率
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out_net, 1), tf.argmax(y_data, 1)), tf.float32))
        # 训练结果保存期
        saver = tf.train.Saver()

        with  tf.Session() as session:
            # 初始化全局变量(不包含站位变量)
            session.run(tf.global_variables_initializer())
            batch_size = 10
            num_batch = len(self.train_x) // batch_size
            # 训练的次数
            for n in range(times):
                # result = session.run(优化器,feed_dict={传递参数})
                for n_b in range(num_batch):
                    batch_x = self.train_x[n_b * batch_size: (n_b + 1) * batch_size]
                    batch_y = self.train_y[n_b * batch_size: (n_b + 1) * batch_size]
                    _, r_loss = session.run([optimizer, loss],
                                            feed_dict={
                                                x_data: batch_x,
                                                y_data: batch_y,
                                                keep_prob_50: 0.5,
                                                keep_prob_75: 0.75})
                    print(f"第{n * num_batch + n_b}训练,损失值: {r_loss}")

            # 获取测试数据的准确率
            acc = accuracy.eval({x_data: self.train_x, y_data: self.train_y, keep_prob_50: 1.0, keep_prob_75: 1.0})
            print(f"after {times} times run: accuracy is {acc}")

            saver.save(session, self.savepath)


    # 预测
    def predict(session, test_x, predict):

        # 开始执行, 取得指定样本下的输出数据
        result = session.run([predict, tf.argmax(predict, 1)],
                             feed_dict={x_data: test_x, keep_prob_50: 1.0, keep_prob_75: 1.0})
        print("预测值:%d\n" % result[1][0])
        return result



