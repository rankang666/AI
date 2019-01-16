#coding=utf-8

import tensorflow as tf
import os
import cv2 as cv
import numpy as np

from __project.prefect.classUse import Utils
from __project.prefect.classUse.FaceDriver import MyDriver as mydriver
from __project.prefect.classUse.ConvNet import CnnNet as myconv

class FaceTP:
    def __init__(self, inputdir, outputdir):
        self.inputdir = inputdir
        self.outputdir = outputdir
        self.pairdirs = [[inputdir,outputdir]]

    # --------------训练------------------

    def getFace(self,imgpath, outdir, IMGSIZE=64):
        ''' 处理图像，进行灰度处理,保存到新的目录 '''
        # 1. 获取原始图像的样本
        filename = os.path.splitext(os.path.basename(imgpath))[0]
        # 2. 处理图像
        face = Utils.dealwithimage(imgpath, IMGSIZE, IMGSIZE)
        # 3. 进行灰度处理,保存到新的目录
        for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            # 3.1 对其中>255与<0的像素进行处理
            facetemp = Utils.relight(face, alpha, bias)
            # 3.2 取得原来的文件名_索引.jpg
            newfilename = f"{filename}_{inx}.jpg"
            cv.imwrite(os.path.join(outdir, newfilename), facetemp)


    def generateFaces(self,pairdirs=None):
        '''
        对每张图片, 构造三个等级的灰度图像样本(目的: 消灭灰度特征)
        :param pairdirs: 输入目录与输出目录
        :return:
        '''
        if pairdirs != None:
            self.pairdirs = pairdirs
        # 获取到输入目录下所有的原始图像(.jpg)的样本,进行灰度处理，保存到新的目录
        # 统计进度
        num = 1
        for inputdir, outputdir in self.pairdirs:
            # 1. 获取inputdir目录下inputname目录的图像
            for name in os.listdir(inputdir):
                inputname, outputname = os.path.join(inputdir, name), os.path.join(outputdir, name)
                if os.path.isdir(inputname):
                    Utils.createdir(outputname)
                    for fileitem in Utils.getFilesInpath(inputname):
                        # 2. 进行灰度处理,保存到新的目录
                        # 取得原来的文件名_索引.jpg
                        print(f"第{num}个文件")
                        num += 1
                        self.getFace(fileitem, outputname)

    # 获取训练集
    def getTrain(self,pairdirs = None):
        if pairdirs != None:
            self.pairdirs = pairdirs
        # 开始对每张图片, 构造三个等级的灰度图像样本(目的: 消灭灰度特征)
        print("--------------------开始构造--------------------")
        self.generateFaces(self.pairdirs)
        print("--------------------构造完成---------------------")
        # 遍历当前目录文件，循环产生样本与对应标签，与标签对应的索引和标签
        outputdir = self.pairdirs[0][1]
        pathLabelPair, index2name = Utils.getFileAndLabel(outputdir)
        print(pathLabelPair, index2name)

        # 产生训练集，读取图像，产生对应标签
        train_x, train_y = Utils.readImage(pathLabelPair)
        train_x = train_x.astype(np.float32) / 255.0
        print("训练集采集完成")

        return  train_x, train_y, index2name


    # 预测
    def predict(faceClassifyFile, trainFacepath, checkpointpath):

        # 使用DataSets的getCamera函数获取设备
        camera, classfier, window = mydriver.getCamera(faceClassifyFile, window="Test")
        print("获取设备")
        # 使用Train的getfileandlabel,获取训练集的样本和标签
        pathlabelpair, index2name = Utils.getFileAndLabel(trainFacepath)
        print("获取标签")

        # 创建对应标签的CNN网络
        predict = myconv.getCnnNet(classNum=len(pathlabelpair))
        print(f"创建{len(pathlabelpair)}网络")

        if tf.train.Saver() != None:
            saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, checkpointpath)
            # 识别的图像索引
            num = 1
            while 1:
                num += 1
                print(f"It's processing {num} image")
                while camera.isOpened():
                    status, img = camera.read()
                    print(f"It's processing {num} image")
                    if not status:
                        break
                    # 把图像转成灰度图像
                    g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    # 使用分类器识别人脸
                    faces = classfier.detectMultiScale(g_img, 1.3, 5)
                    for x, y, w, h in faces:
                        # 人脸抓取
                        face_img = img[x:x + w, y:y + h]
                        face_img = cv.resize(face_img, (64, 64))
                        # 格式化成预测数据
                        test_x = np.array([face_img])
                        # test_x = test_x.astype(np.float32) / 255.0

                        # 调用卷积神经网络进行预测
                        username = myconv.predict(session, test_x, predict)

                        # 显示名称
                        print(index2name)
                        name = index2name[username[1][0]]
                        img = Utils.showChinese(img, window, name, x, y)

                        # 根据矩形区域,标记人脸
                        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        num += 1

                    cv.imshow(window, img)

                    # 按键检测,q键退出
                    key = cv.waitKey(30)
                    if key & 0xFF == ord("q"):
                        break;



        # 使用DataSets的close函数关闭设备
        mydriver.close(camera, window)

