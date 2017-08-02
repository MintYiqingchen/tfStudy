import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

def weight_variable(shape):
    init=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pooling_22(x):
    # 第二个参数为池化层的大小2*2
    # 第三个参数还不太懂
    return tf.nn.max_pool(x,[1,2,2,1],strides=[1,2,2,1],padding='SAME')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784]) 
y_actual=tf.placeholder(tf.float32,[None,10])

W_conv1=weight_variable([5,5,1,32]) # 最后一维的大小不太明白，是随便设置的吗
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))

x_image=tf.reshape(x,[-1,28,28,1]) # 第一维的-1表示还是按照输入图片的个数自动计算

# 第一层卷积和池化 池化后大小变为14*14
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pooling_22(h_conv1)

W_conv2=weight_variable([5,5,32,64])
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))

# 第二层卷积
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pooling_22(h_conv2)

# 全连接层
W_fc1=weight_variable([7*7*64,1024]) # 为什么图片就降维到了7*7
b_fc1=tf.Variable(tf.constant(0.1,shape=[1024]))

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
# 输出层
W_fc2=weight_variable([1024,10])
b_fc2=tf.Variable(tf.constant(0.1,shape=[10]))

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# 优化器
cross_entropy=-tf.reduce_sum(y_actual*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_actual,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch=mnist.train.next_batch(50)
    if i%100==0:
        if(i%2000==0):
            train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_actual:batch[1],keep_prob:0.5})
            print("step %d:training accuracy %g"%(i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_actual:batch[1],keep_prob:0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_actual:mnist.test.labels,keep_prob:0.5}))
