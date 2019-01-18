# -*- coding: utf-8 -*-
# 당뇨병 데이터를 읽어와서 학습

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy=np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data=xy[:, 0:-1]
y_data=xy[:,[-1]]

print(x_data.shape, y_data.shape)

# 총 8개의 X 데이터가 있음
X=tf.placeholder(tf.float32, shape=[None,8])
Y=tf.placeholder(tf.float32, shape=[None,1])

# 입력받을때 8개가 들어와서 1개로나가니까 [8,1]
W=tf.Variable(tf.random_normal([8,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')


# Hypothesis : sigmoid 사용
# 텐서플로우로 직접 구현 하는 법 : tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis=tf.sigmoid(tf.matmul(X,W)+b)

# cost/loss function
cost=-tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

# minimize 는 GradientDescentOptimizer 알고리즘을 사용
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# 정확도 측정 0.5 기준
# tf.case() 는 true / false 를 1.0 / 0.0 으로 반환해준다
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost,train], feed_dict={X:x_data, Y:y_data})
        if step%200 == 0:
            print(step, cost_val)


     # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("Hypothesis", h)
    print("Correct (Y)", c)
    print("Accuracy", a)
