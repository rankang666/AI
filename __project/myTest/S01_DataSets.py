#coding=utf-8

import numpy as np
from cv2 import *
import tensorflow as tf


# 视频采集
# 视频的显示窗体定义
cv2.namedWindow("w_img")
cv2.setWindowTitle("w_img", "Video")
cv2.moveWindow("w_img",0,0)

# 构建一个人脸分类器
# classfier = cv2.CascadeClassifier()
# classfier.load("haarcascade_frontalface_alt2.xml")
classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# 打开摄像头，获取摄像设备
cap =cv2.VideoCapture(0)

# 图像编号
counter = 0
# 循环采集每一帧图像(保存, 显示)
while (cap.isOpened()):
    status, frame = cap.read()
    # 读取是否成功,失败退出
    if not status:
        break
    # 采用TensorFlow框架,截取人脸,并保存成图像库(使用流水编号)
    # 1. 把图像变成灰度(便于人脸区域识别)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. 侦测人脸
    faces = classfier.detectMultiScale(gray_frame,1.2,3)
    # 3. 在视频图像中,标记出人脸
    # 把视频帧图像显示在窗体
    for face in faces:
        x, y, w, h = face
        # 保存图像
        # 取出人脸(人脸图像就是像素矩阵)
        f_img = frame[y:y+h, x:x+w]
        # 保存(先把图像拉成64*64)
        f_img = cv2.resize(f_img,(64,64))
        cv2.imwrite(f"imgs/ran冉康/{counter}.jpg",f_img)

        # 改变索引
        counter += 1
        if counter >= 200:
             break
        # 标记人脸
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 显示采集数据的进度
        cv2.putText(frame,f"Image Number: {counter}",(x,y-30), cv2.FONT_HERSHEY_SIMPLEX,1,255,2)

    if counter >= 200:
        break

    cv2.imshow("w_img", frame)
    # 退出窗体的操作(按键)
    key = cv2.waitKey(30)
    if key & 0xFF == ord("q"):
        # 退出
        break
# 释放摄像头
cap.release()
#释放窗体
cv2.destroyWindow("w_img")