#coding=utf-8
import numpy as np
import cv2
import tensorflow  as tf
import os

from __project.myTest.S02_ConvNet import CnnNet
from __project.myTest import S02_ConvNet



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



# 封装到S02_ConvNet的predict函数中
def predict():
    # 启动摄像头
    cv2.namedWindow("w_img")
    cv2.moveWindow("w_img",0,0)
    cv2.setWindowTitle("w_img","capture face")
    # 启动摄像头
    cap = cv2.VideoCapture(0)
    # 定义分类器
    classfier = cv2.CascadeClassifier("../haarcascade_frontalface_alt2.xml")
    # 获取标签
    pathlabelpair, indextoname = getfileandlabel('../train')

    # 训练结果的目录
    checkpointpath ="../checkpoint/result.ckpt"

    # 获取预测值
    predict = CnnNet(2)
    saver = tf.train.Saver()

    with tf.Session() as session :
        saver.restore(session, checkpointpath)

        while cap.isOpened():
            status, img = cap.read()
            if not status:
                break
            # 把图像转成灰度图像
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用分类器识别人脸
            faces = classfier.detectMultiScale(g_img, 1.2, 3)
            for x, y, w, h in faces:
                # 人脸抓取
                face_img = img[x:x + w, y:y + h]
                face_img = cv2.resize(face_img, (64, 64))
                # 格式化成预测数据
                # 调用卷积神经网络进行预测
                # username = S02_ConvNet.predict(np.array([face_img]),"./checkpoint/result.ckp")
                username = S02_ConvNet.predict(session,np.array([face_img]),predict)
                # 预测成功，在人脸上显示姓名(目录)
                # 根据矩形区域,标记人脸
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # 显示人脸的姓名
                print(indextoname)
                cv2.putText(img, indextoname[username[1][0]], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                # cv2.putText(img, "ran冉康", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

            cv2.imshow("w_img", img)
            # 按键检测,q键退出
            key = cv2.waitKey(30)
            if key & 0xFF == ord("q"):
                break;

    cap.release()
    cv2.destroyWindow("w_img")

# predict()

# 直接预测
def predict1():
    # 启动摄像头
    cv2.namedWindow("w_img")
    cv2.moveWindow("w_img",0,0)
    cv2.setWindowTitle("w_img","capture face")
    # 启动摄像头
    cap = cv2.VideoCapture(0)
    # 定义分类器
    classfier = cv2.CascadeClassifier("D:\\RAN\\PycharmProjects\\AI\\__project\\haarcascade_frontalface_alt2.xml")
    pathlabelpair, indextoname = getfileandlabel('../train')

    # 获取预测值
    predict = CnnNet(2)

    saver = tf.train.Saver()
    checkpointpath = "D:\\RAN\\PycharmProjects\\AI\\__project\\checkpoint\\result.ckpt"

    with tf.Session() as session:

        saver.restore(session, checkpointpath)

        # 开始执行, 取得指定样本下的输出数据
        while cap.isOpened():
            status, img = cap.read()
            if not status:
                break
            # 把图像转成灰度图像
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用分类器识别人脸
            faces = classfier.detectMultiScale(g_img, 1.2, 3)
            for x, y, w, h in faces:
                # 人脸抓取
                face_img = img[x:x + w, y:y + h]
                face_img = cv2.resize(face_img, (64, 64))
                # 格式化成预测数据
                # 调用卷积神经网络进行预测
                # username = S02_ConvNet.predict(np.array([face_img]),"./checkpoint/result.ckp")

                data = np.array([face_img])
                username = session.run([predict,tf.argmax(predict,1)], feed_dict={S02_ConvNet.x_data: data,
                                                           S02_ConvNet.keep_prob_50: 1.0,
                                                           S02_ConvNet.keep_prob_75: 1.0})
                print(username)
                # 预测成功，在人脸上显示姓名(目录)
                # 根据矩形区域,标记人脸
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # 显示人脸的姓名

                from PIL import Image, ImageDraw, ImageFont
                # 读取文件
                # pil_img = Image.open(img,"r")
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # pil_img.show()
                # 生成画笔
                draw = ImageDraw.Draw(pil_img)
                # 第一个参数是字体文件的路径，第二个是字体大小
                font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
                # 第一个参数是文字的起始坐标，第二个需要输出的文字，第三个是字体颜色，第四个是字体类型
                draw.text((x, y-30), '优秀，哈哈', (0, 255, 255), font=font)

                # PIL图片转cv2
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                # 变得可以拉伸 winname 必须要一样，且设置可以拉伸在前面
                cv2.namedWindow('w_img', cv2.WINDOW_NORMAL)


















                # cv2.putText(img, indextoname[username[1][0]], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                # cv2.putText(img, "ran冉康", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

            cv2.imshow("w_img", img)
            # 按键检测,q键退出
            key = cv2.waitKey(30)
            if key & 0xFF == ord("q"):
                break;
        cap.release()
        cv2.destroyWindow("w_img")

predict1()




def predict2():
    # 启动摄像头
    cv2.namedWindow("w_img")
    cv2.moveWindow("w_img",0,0)
    cv2.setWindowTitle("w_img","capture face")
    # 启动摄像头
    cap = cv2.VideoCapture(0)
    # 定义分类器
    classfier = cv2.CascadeClassifier("D:\\RAN\\PycharmProjects\\AI\\__project\\haarcascade_frontalface_alt2.xml")
    pathlabelpair, indextoname = getfileandlabel('../train')

    # 获取预测值
    predict = CnnNet(2)

    saver = tf.train.Saver()
    checkpointpath = "D:\\RAN\\PycharmProjects\\AI\\__project\\checkpoint\\result.ckpt"

    with tf.Session() as session:

        saver.restore(session, checkpointpath)

        # 开始执行, 取得指定样本下的输出数据
        while cap.isOpened():
            status, img = cap.read()
            if not status:
                break
            # 把图像转成灰度图像
            g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 使用分类器识别人脸
            faces = classfier.detectMultiScale(g_img, 1.2, 3)
            for x, y, w, h in faces:
                # 人脸抓取
                face_img = img[x:x + w, y:y + h]
                face_img = cv2.resize(face_img, (64, 64))
                # 格式化成预测数据
                # 调用卷积神经网络进行预测
                # username = S02_ConvNet.predict(np.array([face_img]),"./checkpoint/result.ckp")

                data = np.array([face_img])
                username = session.run([predict,tf.argmax(predict,1)], feed_dict={S02_ConvNet.x_data: data,
                                                           S02_ConvNet.keep_prob_50: 1.0,
                                                           S02_ConvNet.keep_prob_75: 1.0})
                print(username)
                # 预测成功，在人脸上显示姓名(目录)
                # 根据矩形区域,标记人脸
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                # 显示人脸的姓名

                from PIL import Image, ImageDraw, ImageFont
                # 读取文件
                # pil_img = Image.open(img,"r")
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # pil_img.show()
                # 生成画笔
                draw = ImageDraw.Draw(pil_img)
                # 第一个参数是字体文件的路径，第二个是字体大小
                font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
                # 第一个参数是文字的起始坐标，第二个需要输出的文字，第三个是字体颜色，第四个是字体类型
                draw.text((x, y-30), '优秀，哈哈', (0, 255, 255), font=font)

                # PIL图片转cv2
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                # 变得可以拉伸 winname 必须要一样，且设置可以拉伸在前面
                cv2.namedWindow('w_img', cv2.WINDOW_NORMAL)


















                # cv2.putText(img, indextoname[username[1][0]], (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                # cv2.putText(img, "ran冉康", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

            cv2.imshow("w_img", img)
            # 按键检测,q键退出
            key = cv2.waitKey(30)
            if key & 0xFF == ord("q"):
                break;
        cap.release()
        cv2.destroyWindow("w_img")





