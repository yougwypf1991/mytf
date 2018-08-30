import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# 载入数据
# one_hot将标签转换为矩阵形式，间上一课视频
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义每个批次的大小
batch_size = 100
period_epoch = 200

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 28*28]) # None会被batch_size替换
y = tf.placeholder(tf.float32, [None, 10])    #  标签

# 创建一个简单的神经网路
W1 = tf.Variable(tf.random_normal([784, 300]))
b1 = tf.Variable(tf.zeros([300]))
Z1 = tf.matmul(x, W1) + b1
active1 = tf.nn.tanh(Z1)

# 300个隐藏层
W2 = tf.Variable(tf.random_normal([300, 30]))
b2 = tf.Variable(tf.zeros([30]))
Z2 = tf.matmul(active1, W2) + b2
active2 = tf.nn.tanh(Z2)

# 10个隐藏层
W3 = tf.Variable(tf.random_normal([30, 10]))
b3 = tf.Variable(tf.zeros([10]))
Z3 = tf.matmul(active2, W3) + b3
active3 = tf.nn.tanh(Z3)

prediction = tf.nn.softmax(active3)

# 定义二次代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降
train_step = tf.train.GradientDescentOptimizer(5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# equal函数中参数一样的话返回True，否则返回False
# argmax函数是求最大的数是在第几个位置， 1表示按行查找，0表示按列查找
#       prediction就是一个概率。位置相当于就是0-9的标签。
# 结果存放在bool型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
# cast函数将correct_prediction转换Wie浮点型 为1111110000011100011000
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

acc_list = np.empty(period_epoch)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(period_epoch):
        for batch in range(n_batch):
            # 获取一个批次，图像数据在batch_xs，标签在batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys})
        # 每个周期看一下准确率
        acc = sess.run(accuracy, 
                       feed_dict = {x:mnist.test.images, y:mnist.test.labels})
        print('Iter ' + str(epoch) + ', Testing Accuracy ' + str(acc), end = '\r')
        acc_list[epoch] = acc
        
# 画图
axis = np.linspace(0, period_epoch, period_epoch)
plt.figure()
# plt.scatter(epoch)
# r代表画的线为红色，-代表画的线为实线
plt.plot(axis, acc_list, 'r-')
plt.show()