import tensorflow as tf


# 以下是对各种张量tensor的定义
# 定义一个随机数 标量
random_float = tf.random.uniform(shape=())

#  定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))
#  创建一个常量op，1*2矩阵，返回值代表常量op的返回值
matrix1 = tf.constant([[3., 3.]])
#  创建另一个常量op，产生2*1矩阵
matrix2 = tf.constant([[2.], [2.]])
# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])



#  获取张量的属性
#  返回矩阵的形状
print(A.shape)
#  返回数据类型
print(A.dtype)
#  返回矩阵
print(A.numpy())
#  可以加入dtype参数对数据类型进行规定



#  创建矩阵乘法op，把以上两个矩阵作为输入
#  返回值为product

product = tf.matmul(matrix1, matrix2)
#  以上有三个op，两个constant(),一个matmul()


#  tensorflow的不同操作

C = tf.add(A,B)  #计算咯昂个矩阵的和
D = tf.matmul(A,B)  #计算两个矩阵的乘积

# 自动求导机制




