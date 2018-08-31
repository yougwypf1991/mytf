import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# 载入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 定义每个批次的大小
batch_size = 100
period_epoch = 31
test_acc_list = np.empty(period_epoch)
train_acc_list = np.empty(period_epoch)

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 28*28]) # None会被batch_size替换
y = tf.placeholder(tf.float32, [None, 10])    #  标签
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

# 创建一个简单的神经网路
W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev = 0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
Z1 = tf.matmul(x, W1) + b1
L1 = tf.nn.tanh(Z1)
# keep_prob意思是有多少个神经元是工作的
# 1 就是100%的神经元在工作
L1_drop = tf.nn.dropout(L1, keep_prob)


W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev = 0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
Z2 = tf.matmul(L1_drop, W2) + b2
L2 = tf.nn.tanh(Z2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev = 0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
Z3 = tf.matmul(L2_drop, W3) + b3
L3 = tf.nn.tanh(Z3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev = 0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

# 定义二次代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(period_epoch):
        for batch in range(n_batch):
            # 获取一个批次，图像数据在batch_xs，标签在batch_ys
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys,keep_prob:1.0})
        # 每个周期看一下准确率
        test_acc = sess.run(accuracy, 
            feed_dict = {x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, 
            feed_dict = {x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
        
        print('Iter ' + str(epoch) + ', Testing Accuracy ' + str(test_acc), ', Training Accuracy ', str(train_acc), end = '\r')
        test_acc_list[epoch] = test_acc
        train_acc_list[epoch] = train_acc
        
# 画图
axis = np.linspace(0, period_epoch, period_epoch)
plt.figure()
plt.plot(axis, test_acc_list, 'r-')
plt.plot(axis, train_acc_list, 'b-')
plt.show()