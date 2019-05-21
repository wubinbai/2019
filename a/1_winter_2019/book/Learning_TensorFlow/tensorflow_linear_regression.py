import tensorflow as tf
import numpy as np
x_data = np.random.randn(20,1)
w_real = [10]
b_real = 2

noise = np.random.randn(1,20)*0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise




x = tf.placeholder(tf.float32,shape=[None,1])
y_true = tf.placeholder(tf.float32,shape=None)

w = tf.Variable([[0]],dtype=tf.float32,name='weights')
b = tf.Variable(0,dtype=tf.float32,name='bias')
y_pred = tf.matmul(w,tf.transpose(x)) + b
loss = tf.reduce_mean(tf.square(y_true - y_pred))

learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
NUM_STEPS = 10
wb_ = []
with tf.Session() as sess:
    sess.run(init)
    for step in range(NUM_STEPS):
        sess.run(train,{x: x_data, y_true: y_data})
        if (step > -1 ): #% 5 == 0):
            print("step: ", step,' ', sess.run([w,b]))
            wb_.append(sess.run([w,b]))
    print('step: ', 10, ' ', sess.run([w,b]))
