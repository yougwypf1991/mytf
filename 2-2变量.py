import tensorflow as tf

x = tf.Variable([1, 2])
a = tf.Variable([3, 3])

# 减法op
sub = tf.subtract(x, a)
# 加法op
add = tf.add(x, sub)

# tensorflow中的变量必须要初始化
# 初始化所有的变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行初始化
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
    
    
# 创建一个变量，初始化为0
state = tf.Variable(0, name='counter')
# 创建一个op，作用是使state加一
new_value = tf.add(state, 1)
# 赋值op
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('init:', sess.run(state))
    for _ in range(5):
        # update 调用赋值操作，将新的值赋给state
        sess.run(update)
        print(sess.run(state))
