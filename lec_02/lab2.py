#-*- coding: utf-8 -*-
import tensorflow as tf

x_train=[1,2,3]
y_train=[1,2,3]

# random_normal
# 0~1 사이의 정규확률분포 값을 생성해주는 함수
# 원하는 shape 대로 만들어줌 1을 넣으면 1차원 배열을 생성
# Tensorflow에서는 Variable() 이라는 생성자를 사용해서 변수를 생성할 수 있는데 나중에 global_variables_initializer로 초기화해줘야합니다.
W=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

# 가설
hypothesis=x_train*W+b

# reduce_mean 은 평균을 구해주는 메소드
cost=tf.reduce_mean(tf.square(hypothesis-y_train))

# learning rate 와 gradient의 곱한만큼 이동하며 학습하므로 적절한 learing_rate를 주는 것이 중요하다.
# 너무크면 학습이 발산하고 너무 적으면 학습이 너무 오래 걸린다.
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

# 그래프를 사용하려면 세션 객체가 필요해서 객체를 만들어 줌
sess=tf.Session()

# Variable 사용시 global_variables_initializer() 을 해줘야 에러가 안 뜬다.
sess.run(tf.global_variables_initializer())

# 반복문을 돌려서 step 이 20 으로 나누어질때마다 cost, W, b 값을 출력한다. 만들어준 세션 객체를 실행시킨다.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
