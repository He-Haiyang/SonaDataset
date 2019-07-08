import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_dataset():
  df = pd.read_csv("sonar.csv")
  x = df[df.columns[:60]].values
  y = df[df.columns[60]]
  encoder = LabelEncoder()
  encoder.fit(y)
  y = encoder.transform(y)
  Y = one_hot_encode(y)
  print(x.shape)
  return (x, Y)

def one_hot_encode(labels):
  n_labels = len(labels)
  n_unique_labels =  len(np.unique(labels))
  one_hot_encode = np.zeros((n_labels, n_unique_labels))
  one_hot_encode[np.arange(n_labels), labels]  = 1
  return one_hot_encode

X, Y = read_dataset()

X, Y = shuffle(X, Y, random_state = 1)
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.2, random_state=415)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

learning_rate = 0.3
epochs = 1000
cost_history = np.empty(shape = [1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 2
model_path = "C:\Pycharm_workspace\Sonar\Sonar"

n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, shape=[None, n_dim])
y = tf.placeholder(tf.float32, shape=[None, n_class])

weights = {
  "h1": tf.Variable(tf.random.truncated_normal([n_dim, n_hidden_1])),
  "h2": tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2])),
  "h3": tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3])),
  "h4": tf.Variable(tf.random.truncated_normal([n_hidden_3, n_hidden_4])),
  "out": tf.Variable(tf.random.truncated_normal([n_hidden_4, n_class]))
}

bias = {
  "h1": tf.Variable(tf.random.truncated_normal([n_hidden_1])),
  "h2": tf.Variable(tf.random.truncated_normal([n_hidden_2])),
  "h3": tf.Variable(tf.random.truncated_normal([n_hidden_3])),
  "h4": tf.Variable(tf.random.truncated_normal([n_hidden_4])),
  "out": tf.Variable(tf.random.truncated_normal([n_class]))
}

def MLP(x, weights, bias):
  layer_1 = tf.add(tf.matmul(x, weights["h1"]), bias["h1"])
  layer_1 = tf.nn.sigmoid(layer_1)

  layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), bias["h2"])
  layer_2 = tf.nn.sigmoid(layer_2)

  layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), bias["h3"])
  layer_3 = tf.nn.sigmoid(layer_3)

  layer_4 = tf.add(tf.matmul(layer_3, weights["h4"]), bias["h4"])
  layer_4 = tf.nn.relu(layer_4)

  output = tf.add(tf.matmul(layer_4, weights["out"]), bias["out"])
  #output = tf.nn.sigmoid(output)

  return output

init = tf.global_variables_initializer()

saver = tf.train.Saver()

pred = MLP(x, weights, bias)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
  sess.run(init)
  mse_history = []
  accuracy_history = []
  for epoch in range(epochs):
    _, loss_epoch = sess.run([optimizer, loss], feed_dict={x: train_x, y: train_y})
    cost_history = np.append(cost_history, loss_epoch)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    predict_y  = sess.run(pred, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(predict_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy_ = (sess.run(accuracy, feed_dict={x: train_x, y: train_y}))
    accuracy_history.append(accuracy_)
    print("epoch: ", epoch, "-", "loss: ", loss_epoch, "mse: ", mse_, "Training accuracy: ", accuracy_)
  save_path = saver.save(sess, model_path)
  print("Model saved in file: %s"%save_path)

  plt.plot(mse_history, 'r')
  plt.show()
  plt.plot(accuracy_history)
  plt.show()

  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Test accuracy:  ", sess.run(accuracy, feed_dict={x: test_x, y: test_y}))

  test_predict_y = sess.run(pred, feed_dict={x: test_x})
  test_mse = tf.reduce_mean(tf.square(test_predict_y - test_y))
  print("MSE: %.4f"%sess.run(test_mse))

