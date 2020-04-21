import numpy as np
import tensorflow as tf

#  定义数据并进行归一化处理
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())


#  生成tensor张量
X = tf.constant(X)
y = tf.constant(y)

#  生成优化目标变量 初始值设为1
a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num = 10000
#  梯度下降优化器
#  根据计算出的求导结果更新模型参数，从而最小化某一个参数
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

for e in range(num):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5*tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    #  在这里提供了参数 等待更新的变量
    #  zip函数是把不同的变量组合在一起
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
#  可以较为简单的实现线性回归其，总的来说tensorflow确实要方便不少
print(a,b)