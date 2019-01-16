# coding=utf-8
import numpy as np
import cv2 as cv
import os
import logging as log
import __project.prefect.code.p_2ConvNet as myconv

# 保存的图像大小
IMGSIZE = 64


# 1. 对图像做灰度处理(3个灰度)

def createdir(*args):
    ''' 创建处理后的文件保存目录 '''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

def getPaddingSize(shape):
    ''' 获取使图像成为方形的大小 '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest] * 4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def relight(imgsrc, alpha=1, bias=0):
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


def dealwithimage(imgpath, h=64, w=64):
    ''' 处理图像，把图像填充完整 '''
    # 此处是已经处理好的图像
    img = cv.imread(imgpath)
    top, bottom, left, right = getPaddingSize(img.shape[0:2])
    # cv2.copyMakeBorder(src,top, bottom, left, right ,borderType,value)
    # BORDER_REFLICATE　　　  # 直接用边界的颜色填充， aaaaaa | abcdefg | gggg
    # BORDER_REFLECT　　　　  # 倒映，abcdefg | gfedcbamn | nmabcd
    # BORDER_REFLECT_101　　  # 倒映，和上面类似，但在倒映时，会把边界空开，abcdefg | egfedcbamne | nmabcd
    # BORDER_WRAP　　　　  　  # 额。类似于这种方式abcdf | mmabcdf | mmabcd
    # BORDER_CONSTANT　　　　  # 常量，增加的变量通通为value色 [value][value] | abcdef | [value][value][value]
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=[1, 0, 0])
    img = cv.resize(img, (h, w))
    return img

    # 对图像进行处理
    # img = cv.imread(imgpath)
    # faceClassifyPath = "../utils/haarcascade_frontalface_alt2.xml"
    # classfier = cv.CascadeClassifier(faceClassifyPath)
    # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # faces = classfier.detectMultiScale(gray_img, 1.3, 5)
    # n = 0
    # for f_x, f_y, f_w, f_h in faces:
    #     n += 1
    #     face = img[f_y:f_y + f_h, f_x:f_x + f_w]
    #     # may be do not need resize now
    #     face = cv2.resize(face, (64, 64))
    #     face = dealwithimage(face, IMGSIZE, IMGSIZE)
    #     for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
    #         facetemp = relight(face, alpha, bias)
    #         cv2.imwrite(os.path.join(outdir, '%s_%d_%d.jpg' % (filename, n, inx)), facetemp)

def getFace(imgpath, outdir):
    ''' 处理图像，进行灰度处理,保存到新的目录 '''
    # 1. 获取原始图像的样本
    filename = os.path.splitext(os.path.basename(imgpath))[0]
    # 2. 处理图像
    face = dealwithimage(imgpath, IMGSIZE, IMGSIZE)
    # 3. 进行灰度处理,保存到新的目录

    for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
        # 3.1 对其中>255与<0的像素进行处理
        facetemp = relight(face, alpha, bias)
        # 3.2 取得原来的文件名_索引.jpg
        newfilename = f"{filename}_{inx}.jpg"
        cv.imwrite(os.path.join(outdir, newfilename), facetemp)
# 1.1 开始对每张图片, 构造三个等级的灰度图像样本(目的: 消灭灰度特征)
def generateFaces(pairdirs):
    '''
    对每张图片, 构造三个等级的灰度图像样本(目的: 消灭灰度特征)
    :param pairdirs: 输入目录与输出目录
    :return:
    '''
    # 获取到输入目录下所有的原始图像(.jpg)的样本,进行灰度处理，保存到新的目录
    for inputdir, outputdir in pairdirs:
        # 1. 获取inputdir目录下inputname目录的图像
        for name in os.listdir(inputdir):
            inputname, outputname = os.path.join(inputdir,name), os.path.join(outputdir,name)
            if os.path.isdir(inputname):
                createdir(outputname)
                for fileitem in getFilesInpath(inputname):
                    # 2. 进行灰度处理,保存到新的目录
                    # 取得原来的文件名_索引.jpg
                    getFace(fileitem, outputname)





# 2. 获取样本和标签矩阵 以及样本标签索引和样本所属标签


def onehot(numlist):
    ''' 初始化一个(len*(max+1))维的矩阵(行：个数，列：对应标签)，并标签对应位置为1 '''
    b = np.zeros([len(numlist), max(numlist) + 1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()

# 2.1 使用循环结构, 产生样本与标签
def getFileAndLabel(filedir):
    ''' 遍历当前目录文件，循环产生样本与对应标签，与标签对应的索引和标签 '''
    dictdir = dict([[name, os.path.join(filedir, name)] \
                    for name in os.listdir(filedir) \
                    if os.path.isdir(os.path.join(filedir, name))
                    ])
    # for (path, dirnames. _) in os.walk(filedir) for dirname in dirnames
    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))
    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))

def getFilesInpath(filedir):
    ''' 获取当前目录下的所有(.jpg)文件 '''
    suffix = ".jpg"
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith(suffix):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getFilesInpath(os.path.join(path,diritem))

# 2.2 读取数据
def readImage(pathLabelPair):
    ''' 读取数据,获取样本和标签 '''
    imgs = []
    labels = []
    for filePath, label in pathLabelPair:
        for fileitem in getFilesInpath(filePath):
            img = cv.imread(fileitem)
            imgs.append(img)
            labels.append(label)
    name, label = np.array(imgs), np.array(labels)
    return name, label


inputdir = "../image"
outputdir = "../trainfaces"
savepath = "../checkpoint/face.ckpt"
pairdirs = [[inputdir,outputdir]]

# 训练
def train():
    log.debug("generateFace")

    # 开始对每张图片, 构造三个等级的灰度图像样本(目的: 消灭灰度特征)
    print("--------------------开始构造--------------------")
    generateFaces(pairdirs)
    print("--------------------构造完成---------------------")

    # 遍历当前目录文件，循环产生样本与对应标签，与标签对应的索引和标签
    pathLabelPair, index2name  = getFileAndLabel(outputdir)
    print(pathLabelPair, index2name)

    # 产生训练集，读取图像，产生对应标签
    train_x, train_y = readImage(pathLabelPair)
    train_x = train_x.astype(np.float32) / 255.0
    log.debug(f"len of train_x : {train_x.shape}")

    myconv.train(train_x, train_y, savepath)
    log.debug("training is over, please run again")

if __name__ == "__main__":
    train()




