#!/usr/bin/env python3

import os
import tensorflow as tf

import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # set to '' to disable GPU init
# Reduce TensorFlow logs: 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any


train_dir = Path("augmented_directory/images")
validation_dir = Path("augmented_directory/validation/images")
opts = {
    "batch_size": 32,
    "img_height": 256, 
    "img_width": 256,
}
# replace remote tfds load with loading from local directories
ds_train = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=opts["batch_size"],
    image_size=(opts["img_height"], opts["img_width"]),
    shuffle=True,
)
ds_test = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='int',
    batch_size=opts["batch_size"],
    image_size=(opts["img_height"], opts["img_width"]),
    shuffle=False,
)

predict_dataset = ds_test.unbatch().take(5).batch(1)

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
# derive class names from the directory dataset
class_names = ds_train.class_names
print("Class names:", class_names)

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