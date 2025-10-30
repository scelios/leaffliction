#!/usr/bin/env python3

import sys
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # set to '' to disable GPU init
# Reduce TensorFlow logs: 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot
# print("TensorFlow version: {}".format(tf.__version__))
# print("TensorFlow Datasets version: ",tfds.__version__)

ds_preview, info = tfds.load('penguins/simple', split='train', with_info=True)
df = tfds.as_dataframe(ds_preview.take(5), info)
# print(df)
# print(info.features)

class_names = ['Adélie', 'Chinstrap', 'Gentoo']

ds_split, info = tfds.load(
    "penguins/processed", split=['train[:20%]', 'train[20%:]'],
    as_supervised=True, with_info=True)

ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_test, tf.data.Dataset)

# print(info.features)
df_test = tfds.as_dataframe(ds_test.take(5), info)
# print("Test dataset sample: ")
# print(df_test)

df_train = tfds.as_dataframe(ds_train.take(5), info)
# print("Train dataset sample: ")
# print(df_train)

ds_train_batch = ds_train.batch(32)

features, labels = next(iter(ds_train_batch))

# print(features)
# print(labels)

# plt.scatter(features[:,0],
#             features[:,2],
#             c=labels,
#             cmap='viridis')

# plt.xlabel("Body Mass")
# plt.ylabel("Culmen Length")
# plt.show()

# replace model creation to use an Input layer (removes the UserWarning)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu,
                          input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])


predictions = model(features)
predictions[:5]

tf.nn.softmax(predictions[:5])

# print("Prediction: {}".format(tf.math.argmax(predictions, axis=1)))
# print("    Labels: {}".format(labels))

# Use Keras fit/evaluate instead of manual gradient loop

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

num_epochs = 201

# add this callback before calling fit()


class SingleLineProgress(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0.0)
        acc = logs.get("sparse_categorical_accuracy",
                       logs.get("accuracy", 0.0))
        val_loss = logs.get("val_loss", 0.0)
        val_acc = logs.get("val_sparse_categorical_accuracy",
                           logs.get("val_accuracy", 0.0))
        msg = (f"Epoch {epoch+1}/{self.epochs}  "
               f"loss:{loss:.4f}  acc:{acc:.4f}  "
               f"val_loss:{val_loss:.4f}  val_acc:{val_acc:.4f}")
        # print on one line, overwrite previous
        sys.stdout.write(msg + ("\n" if epoch+1 == self.epochs else "\r"))
        sys.stdout.flush()


progress_cb = SingleLineProgress(num_epochs)

# prepare datasets for fit()
train_ds = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)

# set verbose=0 to suppress Keras built-in printing and use the tqdm callback
history = model.fit(
    train_ds,
    epochs=num_epochs,
    validation_data=val_ds,
    verbose=0,
    callbacks=[TqdmCallback(verbose=1)]
)

# extract metrics for plotting
train_loss_results = history.history['loss']
train_accuracy_results = history.history.get(
    'sparse_categorical_accuracy', history.history.get('accuracy', []))

# plot training metrics
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

# evaluate on the test/validation dataset
test_loss, test_acc = model.evaluate(val_ds, verbose=1)
print("Test set accuracy: {:.3%}".format(test_acc))

predict_dataset = tf.convert_to_tensor([
    [0.3, 0.8, 0.4, 0.5,],
    [0.4, 0.1, 0.8, 0.5,],
    [0.7, 0.9, 0.8, 0.4]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
    class_idx = tf.math.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))


# save the model in specified path
keras_model_path = './keras_save.keras'
model.save(keras_model_path)
