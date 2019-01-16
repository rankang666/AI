#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
# activity_functor = lambda x: x if x>0 else 0
# derivative_functor = 1
activity_functor =lambda x: 1.0/(1+np.exp(-x))
derivative_functor = lambda x: x * (1-x)

class ConvlvedLayer:
    def __init__(self,input_size, kernel_size, kernel_num, learning_rate, a_func, d_func):
       '''
       数据初始化
       :param input_size: (h,w,d)
       :param kernel_size: (h,w,d)的深度必须与输入数据一致
       :param kernel_num: 卷积核的个数
       :param learning_rate: float
       :param a_func: 激活函数
       :param d_func: 激活函数导数函数
       '''
       # 数据定义
       self.kernel_num = kernel_num
       self.kernel_size = kernel_size
       self.kernels = np.random.uniform(-0.1, 0.1,
                                        (self.kernel_size[0],
                                         self.kernel_size[1],
                                         self.kernel_size[2],
                                         self.kernel_num))
       # 计算输出大小
       h = input_size[0] - kernel_size[0] + 1
       w = input_size[1] - kernel_size[1] + 1
       self.o_size = (h, w, self.kernel_num)

       self.values = np.zeros(self.o_size, np.float32)

       self.bais = np.random.uniform(-0.01, 0.01, self.o_size)

       self.a_func = a_func
       self.d_func = d_func
       self.learning_rate = learning_rate

    def forward(self, input_data):
        self.input_data = input_data
        # 向前计算卷积
        # 1. 卷积运算 input_data * kernel
        for d in range(self.o_size[2]):
            for y in range(self.o_size[0]):
                for x in range(self.o_size[1]):
                    # 1.1 在input中取一个子块
                    sub_data = self.input_data[
                        y:y + self.kernel_size[0],
                        x:x + self.kernel_size[1],:]
                    # 1.2 卷积运算值
                    self.values[y,x,d] = (sub_data * self.kernels[:,:,:,d]).sum()

        # 2. 加上偏置值
        self.values += self.bais
        # 3. 激活函数
        self.values = self.a_func(self.values)


    def backward(self, delta):
        # print(delta.shape, self.values.shape)
        # 向后更新权重
        # 条件: 下层误差项,输入数据
        # 1. 根据权重矩阵，定义梯度矩阵
        self.w_grads = np.zeros(self.kernels.shape,np.float32)
        self.b_grads = np.zeros(self.values.shape,np.float32)
        # 2. 根据梯度矩阵进行循环，计算每个梯度值
        for num in range(self.kernels.shape[3]):
            for d in range(self.kernels.shape[2]):
                for y in range(self.kernels.shape[0]):
                    for x in range(self.kernels.shape[1]):
                        # 3. 根据梯度位置，按照误差项的大小，在输入区域找到区域
                        self.w_grads[y,x,d,num] = (self.input_data[
                                    y:y+self.values.shape[0],
                                    x:x+self.values.shape[1],
                                    d] * delta[:,:,num]).sum()
                        self.b_grads[y, x, num] = delta[:, :, num].sum()
                        # 4. 算出区域与误差矩阵的叉乘的和
        self.kernels -= self.learning_rate * self.w_grads
        self.bais -= self.learning_rate * self.b_grads
        self.delta = np.zeros(self.input_data.shape,np.float32)
        # 对下层delta补边
        padding_w = (self.input_data.shape[1] + self.kernels.shape[1] - self.values.shape[1] - 1) / 2
        padding_h = (self.input_data.shape[0] + self.kernels.shape[0] - self.values.shape[0] - 1) / 2
        padding_delta = np.array((padding_h,padding_w,self.values.shape[2]), np.float32)
        padding_delta[
            padding_h:padding_h+self.values.shape[0],
            padding_w,padding_w+self.values.shape[1],:] = delta
        # 传递误差项(上一层的误差项)
        for d in range(self.delta.shape[2]):
            for y in range(self.delta.shape[0]):
                for x in range(self.delta.shape[1]):
                    # self.delta[y,x,d] = self.delta[y,x,d+1]
                    # 取区域
                    sub_p = padding_delta[
                            y:y+self.kernels.shape[0],
                            x:x+self.kernels.shape[1],:]
                    self.delta[y,x,d] = (sub_p * self.kernels[:,:,d,:]).sum()

# 测试代码
layer = ConvlvedLayer(
    input_size= (760, 1000, 3),
    kernel_size= (3,3,3),
    kernel_num= 3,
    learning_rate= 0.001,
    a_func= activity_functor,
    d_func= derivative_functor)

# 准备数据
img = plt.imread("bird2.png")
img = img[:,:,0:3]
layer.forward(img)

# plt.imshow(layer.values[:,:,0])
# plt.show()


# 构造一个误差矩阵(因为没有输出层，一般不会使用卷积层做输出层)
# 虚拟一个误差矩阵
d_d = 3
d_h = (img.shape[0]-3) + 1
d_w = (img.shape[1]-3) + 1
delta_size = (d_h, d_w, d_d)
delta = np.ones(delta_size, np.float32)
layer.backward(delta)



'''
ax1=plt.subplot(2,2,1)
ax2=plt.subplot(2,2,2)
ax3=plt.subplot(2,1,2)

plt.sca(ax1)
plt.imshow(layer.values[:,:,0])
plt.sca(ax2)
plt.imshow(layer.values[:,:,1])
plt.sca(ax3)
plt.imshow(layer.values[:,:,2])
plt.show()
'''


