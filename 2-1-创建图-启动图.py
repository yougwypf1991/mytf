import tensorflow as tf

# 创建一个常量op
# 一行两列
m1 = tf.constant([[3, 3]])
# 两行一列
m2 = tf.constant([[2], [3]])

# 创建一个矩阵乘法op，将m1和m2传入
product = tf.matmul(m1, m2)
print(product)

# 定义一个回话，启动默认的图
# with tf.Session() as sess
sess = tf.Session()

# 调用sess的run方法来执行矩阵乘法op
# run方法触发了图中的三个op
result = sess.run(product)
print(result)
sess.close()
