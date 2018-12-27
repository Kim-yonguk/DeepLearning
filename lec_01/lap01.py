# -*- coding: utf-8 -*-
import tensorflow as tf
print(tf.__version__)

hello=tf.constant("hello, Tensor")

sess=tf.Session()

print(sess.run(hello))


3 # a rank 0 tensor; this is a scalar with shape []
[1. ,2., 3.] # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implictly
node3 = tf.add(node1,node2)

print("node1: ",node1, " node2: ",node2)
print("node3: ",node3)



sess=tf.Session()
print("sess.run(node1,node2): ", sess.run([node1,node2]))
print("sess.run(node3): ", sess.run(node3))



a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)

adder_node=a+b
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))


add_and_triple=adder_node*3.
print(sess.run(add_and_triple, feed_dict={a:3,b:4.5}))
