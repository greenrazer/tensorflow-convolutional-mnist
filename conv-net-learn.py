import tensorflow as tf
import csv
import numpy as np

batch_size = 70
epochs = 300

image_width = 28
image_height = 28
channel_nums = 1

num_classes = 10

filter1_size = 5
num_filters_1 = 16

filter2_size = 5
num_filters_2 = 36
conv2_size = 7

fully_connected1_size = conv2_size*conv2_size*num_filters_2
fully_connected2_size = 128;

train_data = []
train_labels = []

test_data = []
test_labels = []

def make_one_hot_vector(num, shape):
  one_hot = np.zeros(shape);
  one_hot.put(num, 1);
  return one_hot;

print("getting data...")

with open('data/MNIST/mnist_train.csv', "rt", encoding="utf-8") as csvfile:
  read = csv.reader(csvfile)
  for row in read:
    train_labels.append(make_one_hot_vector(row.pop(0), num_classes))
    number = [];
    for num in row:
      number.append(num)
    number = [1 if int(x) > 128 else 0 for x in number]
    reshape_num = np.array(number).reshape([image_width, image_height, channel_nums])
    train_data.append(reshape_num)

with open('data/MNIST/mnist_test.csv', "rt", encoding="utf-8") as csvfile:
  read = csv.reader(csvfile)
  for row in read:
    test_labels.append(make_one_hot_vector(row.pop(0), num_classes))
    number = [];
    for num in row:
      number.append(num)
    number = [1 if int(x) > 128 else 0 for x in number]
    reshape_num = np.array(number).reshape([image_width, image_height, channel_nums])
    test_data.append(reshape_num)

print("finished gathering data.")
print("{0} train data".format(len(train_data)))
print("{0} test data".format(len(test_data)))

print("setting up model...")

def make_conv_layer(input_x, filter_size, channels, num_filters):
  conv_filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, channels, num_filters]), dtype=tf.float32)

  conv_biases = tf.Variable(tf.truncated_normal([num_filters]))

  conv_layer = tf.nn.conv2d(input_x,filter=conv_filter, strides=[1,1,1,1], padding="SAME")

  conv_layer = tf.add(conv_layer, conv_biases)

  pool_layer = tf.nn.max_pool(conv_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

  layer = tf.nn.relu(pool_layer)

  return layer;

def make_fully_connected(input_x, size_in, size_out):
  weights = tf.Variable(tf.truncated_normal([size_in, size_out]))

  biases = tf.Variable(tf.truncated_normal([size_out]))

  fully_connected_layer = tf.add(tf.matmul(input_x, weights), biases)

  fully_connected_layer = tf.sigmoid(fully_connected_layer)

  return fully_connected_layer


x = tf.placeholder(tf.float32, [None, image_width, image_height, channel_nums])

conv_layer_1 = make_conv_layer(x, filter1_size, channel_nums,num_filters_1)

conv_layer_2 = make_conv_layer(conv_layer_1, filter2_size, num_filters_1 ,num_filters_2)

flattened_layer = tf.contrib.layers.flatten(conv_layer_2);

fully_connected_layer_1 = make_fully_connected(flattened_layer, fully_connected1_size, fully_connected2_size)

fully_connected_layer_1 = tf.nn.relu(fully_connected_layer_1)

fully_connected_layer_2 = make_fully_connected(fully_connected_layer_1, fully_connected2_size, num_classes)

y_pred = tf.nn.softmax(fully_connected_layer_2)

y_true = tf.placeholder(tf.float32, [None, num_classes])

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fully_connected_layer_2, labels=y_true)

cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer().minimize(cost);

print("finished setting up model.")

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

saver = tf.train.Saver()

print("starting session...")

with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())
  for j in range(0, epochs):
    for i in range(0,int(len(train_data)/batch_size)):

      batch_train_x, batch_train_y = next_batch(batch_size, train_data, train_labels)

      loss, _ = sess.run([cost, optimizer], feed_dict={x:batch_train_x, y_true:batch_train_y})

    batch_test_x, batch_test_y = next_batch(batch_size, test_data, test_labels)

    y_prediction =  sess.run(y_pred, feed_dict={x:batch_test_x})

    y_pred_max = tf.argmax(y_prediction, axis=1).eval()
    y_true_max = tf.argmax(batch_test_y, axis=1).eval()

    accTrue = 0
    accAll = 0
    for k, q in zip(y_pred_max, y_true_max):
      if(k == q):
        accTrue = accTrue + 1
      accAll = accAll + 1

    print("epoch {0} accuracy is {1}".format(j,(accTrue/accAll)*100))
    save_path = saver.save(sess, "/tmp/model.ckpt", global_step=j)

print("session finished.")
