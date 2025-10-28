#!/usr/bin/env python3

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # set to '' to disable GPU init
# Reduce TensorFlow logs: 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt

ds_split, info = tfds.load("penguins/processed", split=['train[:20%]', 'train[20%:]'], as_supervised=True, with_info=True)

ds_test = ds_split[0]
ds_train = ds_split[1]

predict_dataset = tf.convert_to_tensor([
    [0.3, 0.8, 0.4, 0.5,],
    [0.4, 0.1, 0.8, 0.5,],
    [0.7, 0.9, 0.8, 0.4]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
class_names = ['Adélie', 'Chinstrap', 'Gentoo']

keras_model_path = './keras_save.keras'


# load the model back
restored_keras_model = tf.keras.models.load_model(keras_model_path)
# compile the model (required to make predictions)
restored_keras_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
restored_keras_model.fit(ds_train.batch(32), epochs=1)
print("Model loaded from ./penguin_model_demo.kera")
predictions = restored_keras_model(predict_dataset, training=False)
for i, logits in enumerate(predictions):
  class_idx = tf.math.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction (from loaded model): {} ({:4.1f}%)".format(i, name, 100*p))