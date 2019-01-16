#coding=utf-8

import numpy as np
import os
import cv2 as cv

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

def readText(text):
    ''' 使用语音读取 '''
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def createdir(*args):
    ''' 创建处理后的文件保存目录 '''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

def getFilesInpath(filedir):
    ''' 获取当前目录下的所有(.jpg)文件 '''
    suffix = ".jpg"
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith(suffix):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getFilesInpath(os.path.join(path,diritem))

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
def getPaddingSize(shape):
    ''' 获取使图像成为方形的大小 '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest] * 4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

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
def onehot(numlist):
    ''' 初始化一个(len*(max+1))维的矩阵(行：个数，列：对应标签)，并标签对应位置为1 '''
    b = np.zeros([len(numlist), max(numlist) + 1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()


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

def showChinese(img,windowname,text,x=0,y=30):
    ''' 显示中文 '''
    from PIL import Image, ImageDraw, ImageFont
    # 读取文件
    # pil_img = Image.open(img,"r")
    pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # pil_img.show()
    # 生成画笔
    draw = ImageDraw.Draw(pil_img)
    # 第一个参数是字体文件的路径，第二个是字体大小
    font = ImageFont.truetype('simhei.ttf', 30, encoding='utf-8')
    # 第一个参数是文字的起始坐标，第二个需要输出的文字，第三个是字体颜色，第四个是字体类型
    draw.text((x, y - 30), text, (0, 255, 255), font=font)

    # PIL图片转cv2
    img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    # 变得可以拉伸 winname 必须要一样，且设置可以拉伸在前面
    cv.namedWindow(windowname, cv.WINDOW_NORMAL)
    return img
