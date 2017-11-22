import tensorflow as tf
# from dataset import Cifar

def conv(x, W, name):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
	
def max_pooling(x, k, name):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1],padding='SAME', name=name)
# 归一化操作
def norm(name, x, size=4):
	
# 超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

dropout = 0.8

input_x = tf.placeholder(tf.float32,[None,784])
input_y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

# conv1
Weight_1 = tf.Variable(tf.truncated_normal([5,5,3,32]))
bias_1 = tf.Variable(tf.constant(0.1, shape = [32]))