#coding=utf-8
from __project.prefect.classUse.ConvNet import CnnNet
from __project.prefect.classUse.FaceDriver import MyDriver
from __project.prefect.classUse.FaceTP import FaceTP
import os
# 人脸分类器文件
faceClassifyPath = "../utils/haarcascade_frontalface_alt2.xml"
# 数据保存路径

inputdir = "../image"
outputdir = "../trainfaces"
savepath = "../checkpoint/face1.ckpt"



if __name__ == "__main__":
    # 采集人脸数据
    # driver = MyDriver(inputdir,faceClassifyPath,counter=200)
    # driver.getFace()

    # 训练数据集，将结果保存
    # faceTrain = FaceTP(inputdir, outputdir)
    # train_x, train_y, index2name = faceTrain.getTrain()
    # cnn = CnnNet(train_x, train_y, savepath)
    # cnn.train()

    FaceTP.predict(faceClassifyPath, outputdir, savepath)










