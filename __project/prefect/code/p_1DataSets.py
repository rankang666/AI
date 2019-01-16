#coding=utf-8

import numpy as np
import cv2  as cv
import os


def createDir(*args):
    ''' 创建文件路径 '''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)


def relight(imgsrc, alpha=1,bias=0):
    ''' 降低光线亮度的影响 '''
    # print(imgsrc)
    # 把图像类型转换为小数
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    # 对其中>255与<0的像素进行处理
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    # print(imgsrc)
    return imgsrc



def getCamera(faceClassifyPath,windowname="my_img"):
    ''' 获取摄像设备，分类器 '''
    # 1. 视频的显示窗体定义
    window= windowname
    cv.namedWindow(window)
    cv.setWindowTitle(window, windowname)
    cv.moveWindow(window, 0, 0)
    # 2. 打开摄像头，获取摄像设备
    camera = cv.VideoCapture(0)
    # 3. 构建一个人脸分类器
    classfier = cv.CascadeClassifier(faceClassifyPath)
    return camera, classfier, window


def close(camera,window="my_img"):
    ''' 关闭摄像设备 '''
    #  释放摄像头
    camera.release()
    #  释放窗体
    cv.destroyWindow(window)

def collection(camera, classfier,outdir,counter=100,IMGSIZE=64,window="my_img"):
    ''' 采集图像 '''
    # 图像编号
    num = 1
    # 4. 循环采集每一帧图像(保存, 显示)
    while (camera.isOpened()):
        status, frame = camera.read()
        # 读取是否成功,失败退出
        if not status:
            break
        # 4.1 把图像变成灰度(便于人脸区域识别)
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 4.2 侦测人脸
        faces = classfier.detectMultiScale(gray_frame, 1.3, 5)
        # 4.3 在视频图像中,标记出人脸
        # 把视频帧图像显示在窗体
        for face in faces:
            x, y, w, h = face
            # 4.3.1 取出人脸(人脸图像就是像素矩阵)
            face_img = frame[y:y + h, x:x + w]
            # 4.3.2 保存图像
            # 4.3.2.1 保存(先把图像拉成64*64)
            face_img = cv.resize(face_img, (IMGSIZE, IMGSIZE))
            # 4.3.2.2 保存(先把图像拉成64*64)
            face_img = relight(face_img, np.random.uniform(0.5, 1.5), np.random.randint(-50, 50))
            # 4.3.2.3 保存图像到outdir+num.jpg里面
            cv.imwrite(os.path.join(outdir, str(num) + ".jpg"), face_img)

            # 4.3.3 显示采集数据的进度
            cv.putText(frame, f"Image Number: {num}", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            # 4.3.4 标记人脸
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 4.3.5 显示图像
            cv.imshow(window, frame)
            # 4.3.6 改变索引
            num += 1
        # 4.3.7 退出窗体的操作(按键)
        key = cv.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            # 退出
            break
        # 4.3.7 条件退出
        if num >= counter:
            break

def getFace(outdir,faceClassifyPath):
    ''' 采集人脸数据 '''
    # 创建保存路径
    createDir(outdir)
    # 获取设备
    camera, classfier, window = getCamera(faceClassifyPath,windowname="Data")
    # 采集图像
    collection(camera,classfier,outdir,counter=100,IMGSIZE=64,window=window)
    # 关闭设备
    close(camera,window=window)

# 人脸分类器文件
faceClassifyPath = "../utils/haarcascade_frontalface_alt2.xml"

# 数据保存路径
outdir = "../image/ran2"

# 测试
if __name__ == "__main__":
    getFace(outdir,faceClassifyPath)











# 通过摄像头采集人脸数据
def getFaceFromCamera():
    # 一. 创建文件路径
    createDir(outdir)

    # 二. 视频采集
    # 1. 视频的显示窗体定义
    cv.namedWindow("my_img")
    cv.setWindowTitle("my_img", "DataSets")
    cv.moveWindow("my_img", 0, 0)
    # 2. 打开摄像头，获取摄像设备
    camera = cv.VideoCapture(0)
    # 3. 构建一个人脸分类器
    classfier = cv.CascadeClassifier(faceClassifyPath)
    # 图像编号
    num = 1
    # 4. 循环采集每一帧图像(保存, 显示)
    while (camera.isOpened()):
        status, frame = camera.read()
        # 读取是否成功,失败退出
        if not status:
            break
        # 4.1 把图像变成灰度(便于人脸区域识别)
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 4.2 侦测人脸
        faces = classfier.detectMultiScale(gray_frame, 1.3, 5)
        # 4.3 在视频图像中,标记出人脸
        # 把视频帧图像显示在窗体
        for face in faces:
            x, y, w, h = face
            # 4.3.1 取出人脸(人脸图像就是像素矩阵)
            face_img = frame[y:y + h, x:x + w]
            # 4.3.2 保存图像
            # 4.3.2.1 保存(先把图像拉成64*64)
            face_img = cv.resize(face_img, (IMGSIZE, IMGSIZE))
            # 4.3.2.2 保存(先把图像拉成64*64)
            face_img = relight(face_img, np.random.uniform(0.5,1.5),np.random.randint(-50,50))
            # 4.3.2.3 保存图像到outdir+num.jpg里面
            cv.imwrite(os.path.join(outdir,str(num)+".jpg"), face_img)

            # 4.3.3 显示采集数据的进度
            cv.putText(frame, f"Image Number: {num}", (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            # 4.3.4 标记人脸
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 4.3.5 显示图像
            cv.imshow("my_img", frame)
            # 4.3.6 改变索引
            num += 1
        # 4.3.7 退出窗体的操作(按键)
        key = cv.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            # 退出
            break
        # 4.3.7 条件退出
        if num >= counter:
            break

    # 5. 释放摄像头
    camera.release()
    # 6. 释放窗体
    cv.destroyWindow("my_img")

# getFaceFromCamera()

