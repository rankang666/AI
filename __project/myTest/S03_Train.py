# coding=utf-8
import numpy as np
import cv2
import os

# 准备数据: 样本, 标签
# 1. 对图像做灰度处理(3个灰度)
import __project.myTest.S02_ConvNet as myconv


def relight(imgsrc, alpha, bias):
    # 把图像类型转换为小数
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    # 对其中>255与<0的像素进行处理
    imgsrc[imgsrc > 255] = 255
    imgsrc[imgsrc < 0] = 0
    imgsrc.astype(np.uint8)
    return imgsrc

# 开始对每张图片, 构造三个等级的灰度图像样本(目的: 消灭灰度特征)
def generateImages():
    # 1. 获取原始图像的样本
    # oldpath = "../imgs/gu"
    # newpath = "../train/gu"
    oldpath = "../imgs/ran冉康"
    newpath = "../train/ran冉康"
    for imagefile in os.listdir(oldpath):
        # 2. 进行灰度处理,保存到新的目录
        # 取得原来的文件名_索引.jpg
        basefilename = os.path.basename(imagefile)
        basename = os.path.splitext(basefilename)[0]
        imgsrc = cv2.imread(os.path.join(oldpath,imagefile))
        print(imagefile)
        for inx,(alpha,bais) in enumerate([[1,1],[1,50],[0.5,0]]):
            newfilename = "%s_%d.jpg"%(basename,inx)
            img = relight(imgsrc,alpha,bais)
            cv2.imwrite(os.path.join(newpath,newfilename),img)


def readimage(pairpathlabel):
    '''read image to list'''
    imgs = []
    labels = []
    for filepath, label in pairpathlabel:
        for fileitem in myconv.getfilesinpath(filepath):
            img = cv2.imread(fileitem)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)




def readImageData():
    # data2label = (("gu",[1,0]),("ran冉康",[1,0])) # 建议写一个函数自动生成这个结构
    # data2label = (("ran冉康", [1,0]),)
    # xdata = []
    # ydata = []
    # # 循环上述结构,产生样本与标签
    # for num in range(len(data2label)):
    #     name, label = data2label[num]
    # 根据目录名, 遍历文件,读取图像,产生数据
    # for filename in os.listdir(f"../train/{name}"):
    #     # print(filename)
    #     img = cv2.imread(f"../train/{name}/{filename}")
    #     xdata.append(img)
    #     ydata.append(label)
    # return (np.array(xdata), np.array(ydata))
    pathlabelpair, indextoname = myconv.getfileandlabel("../train")
    name, label = readimage(pathlabelpair)
    return (name, label)





# 调用
# generateImages()


# # 读取数据
(xdata, ydata) = readImageData()
# print(len(xdata), len(ydata))
print(xdata.shape, ydata.shape)

myconv.train(xdata, ydata, "../checkpoint/result.ckpt")