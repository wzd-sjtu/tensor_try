import tensorflow as tf

#  演示tensorflow的自动求导机制

#  变量variable表示维护状态
#  这里的x是一个变量
x = tf.Variable(initial_value=3.)

with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y, x)
print([y, y_grad])

#  实例 用tensorflow对各种类型求导
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)

#  tf.square 表示对张量中的每一个元素求平方
#  tf.reduce_sum 表示对输入张量的所有元素求和
with tf.GradientTape() as tape:
    L = 0.5*tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
#  计算L(w, b) 关于w, b的偏导数
w_grad, b_grad = tape.gradient(L, [w, b])