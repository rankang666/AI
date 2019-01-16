#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
# 显示窗体分区
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,2,3)
# 图像读取
img1 = plt.imread("bird1.png")
img2 = plt.imread("bird2.png")
# print(img1)
# print(img1.shape)
# img1[:,:,3] = 0.2
# 图像处理
# 核心数据
# kernel = [
#     [0,0,0,0,0],
#     [0, 0,0,0,0],
#     [-1,-1,2,0,0],
#     [0,0,0,0,0],
#     [0,0,0,0,0]
# ]

kernel = np.array([
    [1.0/16,1.0/8,1.0/16],
    [1.0 / 8, 1.0 / 4, 1.0 / 8],
    [1.0 / 16, 1.0 / 8, 1.0 / 16],
])

# 计算特征图像的大小
# 原图的大小
w = img1.shape[1]
h = img1.shape[0]
# 补边(补多大: kernel为3,补1, 5,补2)
padding_width = 1
# 分配新矩阵
nimg = np.zeros((img1.shape[0]+padding_width*2,img1.shape[1]+padding_width*2,4),dtype=np.float32)
# 拷贝数据
nimg[
    padding_width:padding_width+img1.shape[0],
    padding_width:padding_width+img1.shape[1],:
] = img1

# 卷积核的大小
# kw = 5
# kh = 5
kw = 3
kh = 3
# 步长
s = 1
# 分配特征图的空间
cw = (w - kw) + 1
ch = (h - kh) + 1
print(f"原图的大小:({w,h}),特征图大小({cw, ch})")
cimg = np.zeros((ch, cw, nimg.shape[2]),np.float32)
cimg[:,:,3] = 1
# 计算特征图的每个像素
for d in range(img1.shape[2]-1):
    for y in range(ch):
        for x in range(cw):
            # 计算每个像素的图像
            # 计算原图的区域，得到一个深度为4的矩阵
            subimg = nimg[y:y+kh,x:x+kw,d]
            # 做卷积运算
            cimg[y,x,d] =(subimg * kernel).sum() if (subimg* kernel).sum() > 0 else 0
            # print(cimg[y,x,d])

print(cimg)

# 图像显示
plt.sca(ax1)
plt.imshow(img1)

plt.sca(ax2)
img2[:,:,2] = 0
# plt.imshow(img2[:,:,2])
plt.imshow(img2)

plt.sca(ax3)
plt.imshow(cimg)
plt.show()