import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
# 在定义域(-0.5, 0.5)之间均匀的生成200个随机点, 得到200行，一列的数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成干扰项
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeloader
x = tf.placeholder(tf.float32, [None, 1])  # 行不确定，只有1列
y = tf.placeholder(tf.float32, [None, 1])

# 构建一个简单的神经网络，中间层
Weight_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 中间层的带权输入
Wx_plus_b_L1 = tf.matmul(x, Weight_L1) + biases_L1
# 激活函数, 得到中间层的输出
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义输出层
Weight_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
# 输出层的带权输入
Wx_plus_b_L2 = tf.matmul(L1, Weight_L2) + biases_L2
# 输出
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        # 训练train_step 和样本
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
        
    # 获取预测值, 需要传入样本
    prediction_value = sess.run(prediction, feed_dict = {x:x_data})
    
    # 画图
    plt.figure()
    # 打印出样本点
    plt.scatter(x_data, y_data)
    # r代表画的线为红色，-代表画的线为实线
    plt.plot(x_data, prediction_value, 'r-', linewidth = 5)
    plt.show()