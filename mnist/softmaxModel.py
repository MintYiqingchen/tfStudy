import input_data
import tensorflow as tf

# 接受任意数量的784的展开的向量
x=tf.placeholder("float",[None,784])
W=tf.Variable(tf.random_uniform([784,10]))
b=tf.Variable(tf.random_uniform([10]))

# 回归模型
y_predict=tf.nn.softmax(tf.matmul(x,W)+b)

# 计算交叉熵
y_actual=tf.placeholder("float",[None,10])
cross_entropy=-tf.reduce_sum(y_actual*tf.log(y_predict))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 测试准确度
correct_prediction=tf.equal(tf.argmax(y_actual,1),tf.argmax(y_predict,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))

init=tf.global_variables_initializer()

mnist_train=input_data.Imagedata()
mnist_train.read_data("train-images.idx3-ubyte","train-labels.idx1-ubyte")
print("训练集读取完毕")
mnist_test=input_data.Imagedata()
mnist_test.read_data("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte")
print("测试集读取完毕")

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs,batch_ys=mnist_train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_actual:batch_ys})
        if(i%100==0):
            print("accuracy:",sess.run(accuracy, feed_dict={x: mnist_test.images, y_actual: mnist_test.labels}))
print("训练完毕")


