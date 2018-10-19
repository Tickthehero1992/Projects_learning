import tensorflow as tf 
import numpy as np 

x=tf.constant([[1],[2],[3],[4]], dtype=tf.float32) #inputs
y_true = tf.constant([[0],[-1],[-2],[-3]], dtype= tf.float32)#outputs for inputs x

linear_model = tf.layers.Dense(units=1) # model

y_pred = linear_model(x) #output of model

#try predict 
sess=tf.Session() #tf session
init = tf.global_variables_initializer() #initialization of variables
sess.run(init) #start of session

print (sess.run(y_pred))# bad prediction need losses

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred) #losses quared error

print (sess.run(loss)) # need optimize

optimizer= tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

for i in range(2000):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print (sess.run(y_pred))