#coding=utf-8

import numpy as np
import cv2  as cv
import os

from __project.prefect.classUse import Utils


class MyDriver:
    def __init__(self,outdirs,faceClassifyPath, window="DataSets",counter= 100,imgsize = 64):
        '''
        采集人脸数据设备
        :param outdirs: 数据保存路径
        :param faceClassifyPath: 人脸分类器文件
        :param windowname: 创建窗口名称
        :param counter: 采集图像数目
        :param imgsize: 采集图像大小
        '''
        self.outdirs = outdirs
        self.faceClassifyPath = faceClassifyPath
        self.window = window
        self.counter = counter
        self.imgsize = imgsize



    # 获取设备
    def getCamera(faceClassifyPath, window="DataSets"):
        ''' 获取摄像设备，分类器 '''
        cv.namedWindow(window)
        cv.setWindowTitle(window, window)
        cv.moveWindow(window, 0, 0)
        # 2. 打开摄像头，获取摄像设备
        camera = cv.VideoCapture(0)
        # 3. 构建一个人脸分类器
        classfier = cv.CascadeClassifier(faceClassifyPath)
        return camera, classfier, window

    # 采集图像
    def collection(self, camera = None, classfier= None, outdirs = None, window=None):
        if camera != None:
            self.camera = camera
        if classfier != None:
            self.classfier = classfier
        if outdirs != None:
            self.outdirs = outdirs
        if window != None:
            self.windowname = window

        ''' 采集图像 '''
        # 图像编号
        num = 1
        # 4. 循环采集每一帧图像(保存, 显示)
        while (self.camera.isOpened()):
            status, frame = self.camera.read()
            # 读取是否成功,失败退出
            if not status:
                break
            # 4.1 把图像变成灰度(便于人脸区域识别)
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 4.2 侦测人脸
            faces = self.classfier.detectMultiScale(gray_frame, 1.3, 5)
            # 4.3 在视频图像中,标记出人脸
            # 把视频帧图像显示在窗体
            for face in faces:
                x, y, w, h = face
                # 4.3.1 取出人脸(人脸图像就是像素矩阵)
                face_img = frame[y:y + h, x:x + w]
                # 4.3.2 保存图像
                # 4.3.2.1 保存(先把图像拉成64*64)
                face_img = cv.resize(face_img, (self.imgsize, self.imgsize))
                # 4.3.2.2 保存(先把图像拉成64*64)
                face_img = Utils.relight(face_img, np.random.uniform(0.5, 1.5), np.random.randint(-50, 50))
                # 4.3.2.3 保存图像到outdir+num.jpg里面
                cv.imwrite(os.path.join(self.outdirs.split(",")[0], str(num) + ".jpg"), face_img)

                # 4.3.3 显示采集数据的进度
                cv.putText(frame, f"Image Number: {num}", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                # 4.3.4 标记人脸
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # 4.3.5 显示图像
                cv.imshow(self.window, frame)
                # 4.3.6 改变索引
                num += 1
            # 4.3.7 退出窗体的操作(按键)
            key = cv.waitKey(30) & 0xFF
            if key == 27 or key == ord("q"):
                # 退出
                break
            # 4.3.7 条件退出
            if num > self.counter:
                break

    # 关闭设备
    def close(camera, window):
        ''' 关闭摄像设备 '''
        #  释放摄像头
        camera.release()
        #  释放窗体
        cv.destroyWindow(window)

    # 采集人脸数据
    def getFace(self):
        ''' 采集人脸数据 '''
        # 创建保存路径
        Utils.createDir(self.outdirs)
        # 获取设备
        self.camera, self.classfier, self.window = MyDriver.getCamera(self.faceClassifyPath, window="DataSets")
        # 采集图像
        self.collection()
        # 关闭设备
        MyDriver.close(self.camera,self.window)


        # # 创建保存路径
        # self.createDir(self.outdirs)
        # # 获取设备
        # camera, classfier, window = self.getCamera(self.faceClassifyPath, window="Data")
        # # 采集图像
        # self.collection(camera, classfier, self.outdirs, window=window)
        # # 关闭设备
        # self.close(camera, window=window)








