import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])
y_actual=tf.placeholder(tf.float32,[None,10])

