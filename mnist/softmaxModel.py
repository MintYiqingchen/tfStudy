import input_data
import tensorflow as tf

# 接受任意数量的784的展开的向量
x=tf.placeholder("float",[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

# 回归模型
y=tf.nn.softmax(tf.matmul(x,W)+b)

# 计算交叉熵
y_=tf.placeholder("float",[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init=tf.tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

mnist_train=input_data.Imagedata()
mnist_train.read_data("train-images.idx3-ubyte","train-labels.idx1-ubyte")
print("训练集读取完毕")

for i in range(1000):
    batch_xs,batch_ys=mnist_train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

print("训练完毕")
mnist_test=input_data.Imagedata()
mnist_test.read_data("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte")

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))


print(sess.run(accuracy,feed_dict={x:mnist_test.images,y_:mnist_test.labels}))


