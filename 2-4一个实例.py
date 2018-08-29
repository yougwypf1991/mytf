import tensorflow as tf
import numpy as np

#使用numpy生成100个随机点
x_data = np.random.rand(100)
# y_data为样本值，即真实值
y_data = x_data*0.1 + 0.2

# 构造一个线性的模型
b = tf.Variable(0.)
k = tf.Variable(0.)
# 预测值
y = k*x_data + b

# 定义一个二次代价函数,reduce_mean函数为求平均值的函数
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个梯度下降法来进行训练的优化器, 参数为学习率
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 迭代两百次
    for step in range(201):
        # 最小化loss
        sess.run(train)
        if step % 20 == 0:
            print(step, ':', sess.run([k, b]))