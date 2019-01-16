#coding=utf-8
import numpy as np
import cv2 as cv
import tensorflow  as tf
import os
import __project.prefect.code.p_1DataSets as mydriver
import __project.prefect.code.p_2ConvNet as myconv
import __project.prefect.code.p_3Train as mytrain



# 人脸分类器文件
faceClassifyFile = "../utils/haarcascade_frontalface_alt2.xml"
# 训练集处理后的输出目录
trainFacepath = "../trainfaces"
# 训练结果的目录
checkpointpath = "../checkpoint/face.ckpt"

# 预测
def predict():

    # 使用DataSets的getCamera函数获取设备
    camera, classfier, window = mydriver.getCamera(faceClassifyFile, windowname="Test")
    print("获取设备")
    # 使用Train的getfileandlabel,获取训练集的样本和标签
    pathlabelpair, indextoname = mytrain.getFileAndLabel(trainFacepath)
    print("获取标签")

    # 创建对应标签的CNN网络
    predict = myconv.CnnNet(len(pathlabelpair))
    print(f"创建{len(pathlabelpair)}网络")

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, checkpointpath)
        # 识别的图像索引
        num = 1
        while 1:
            if(num >= 10000):
                print("times is too many, please restart!")
                break
            else:
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
                        test_x = test_x.astype(np.float32) / 255.0

                        # 调用卷积神经网络进行预测
                        username = myconv.predict(session, np.array([face_img]), predict)
                        # print(username)
                        # 预测成功，在人脸上显示姓名(目录)
                        # 显示人脸的姓名
                        print(indextoname)
                        # cv.putText(img, indextoname[username[1][0]], (x, y - 30), cv.FONT_HERSHEY_SIMPLEX, 2,
                        #            (0, 255, 0),
                        #            1)
                        name = indextoname[username[1][0]]
                        img = showChinese(img,window,name,x,y)

                        # 根据矩形区域,标记人脸
                        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        num += 1
                        # cv2.putText(img, "ran冉康", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)

                    cv.imshow(window, img)

                    # 按键检测,q键退出
                    key = cv.waitKey(30)
                    if key & 0xFF == ord("q"):
                        break;

    # 使用DataSets的close函数关闭设备
    mydriver.close(camera,window)



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


if __name__ == "__main__":
    predict()
