# -*- coding: utf-8 -*-
# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# x_data : 4개의 변수로 구성
# y_data : one-hot 방식으로 3가지 label
# [1, 0, 0] = 첫 번째
# [0, 1, 0] = 두 번째
# [0, 0, 1] = 세 번째
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

# label의 개수 = y_data 분류 개수
nb_classes = 3


X=tf.placeholder(tf.float32, shape=[None,4])
Y=tf.placeholder(tf.float32, shape=[None,3])


# x_data의 변수가 4개이므로, W도 4개 / y_label의 분류가 3개이므로, binary classification 3번
W=tf.Variable(tf.random_normal([4,nb_classes]), name="weight")
b=tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

# cost/loss function : cross entropy
# axis = 1는 matmul이 아닌 같은 element의 곱을 의미
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 ==0:
            print(step, sess.run(cost,feed_dict={X:x_data, Y:y_data}))





# Test
    # tf.arg_max() = one-hot encoding으로 가장 큰 값의 index를 return
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, "\n결과:", sess.run(tf.arg_max(a, 1)))
    # [[3.48425168e-03   9.96506214e-01   9.58935289e-06]]
    # 결과: [1]

    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, "\n결과:", sess.run(tf.arg_max(all, 1)))
    # [[2.99357832e-03   9.96996760e-01   9.61958904e-06]
    #  [8.89271736e-01   9.94489938e-02   1.12793017e-02]
    # [9.41215550e-09
    # 3.29720846e-04
    # 9.99670267e-01]]
    # 결과: [1 0 2]
