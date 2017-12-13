from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.transpose(tf.reshape(features["x"], [-1, 3, 32, 32]), perm=[0,2,3,1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Normalization Layer #1
  norm1 = tf.layers.batch_normalization(inputs=pool1)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=norm1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  norm2 = tf.layers.batch_normalization(inputs=pool2)

  # Dense Layer
  pool2_flat = tf.reshape(norm2, [-1, 8 * 8 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main(unused_argv):
  # Load training and eval data
  cifar10_1 = unpickle(r'CIFAR-10/cifar-10-batches-py/data_batch_1')
  cifar10_2 = unpickle(r'CIFAR-10/cifar-10-batches-py/data_batch_2')
  cifar10_3 = unpickle(r'CIFAR-10/cifar-10-batches-py/data_batch_3')
  cifar10_4 = unpickle(r'CIFAR-10/cifar-10-batches-py/data_batch_4')
  cifar10_5 = unpickle(r'CIFAR-10/cifar-10-batches-py/data_batch_5')
  cifar10_test = unpickle(r'CIFAR-10/cifar-10-batches-py/test_batch')

  train_data = np.vstack((cifar10_1[b'data'], cifar10_2[b'data'], cifar10_3[b'data'], cifar10_4[b'data'], cifar10_5[b'data'])).astype(np.float32)
  train_labels = np.asarray(np.hstack((cifar10_1[b'labels'], cifar10_2[b'labels'], cifar10_3[b'labels'], cifar10_4[b'labels'], cifar10_5[b'labels'])), dtype = np.int32)

  eval_data = cifar10_test[b'data'].astype(np.float32)
  eval_labels = np.asarray(cifar10_test[b'labels'], dtype = np.int32)

  # Create the Estimator
  cifar_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/cifar_convnet_model2")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  cifar_classifier.train(
    input_fn=train_input_fn,
    steps=10000,
    hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = cifar_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
