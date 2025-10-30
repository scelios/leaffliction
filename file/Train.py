#!/usr/bin/env python3

# what do i need to do? 

# get my images

# seprate validation images

# augment images

# define setting for the training

# training ....

# compare with validation set, obtain accuracy score!

# save the results

import argparse
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras as ks

def train(epochs, train_dir: Path, validation_dir: Path):
    opts = {
        "batch_size": 32,
        "img_height": 256, 
        "img_width": 256,
    }

    train_ds : Any = ks.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=opts["batch_size"],
        image_size=(opts["img_height"], opts["img_width"]),
    )

    validation_ds : Any = ks.utils.image_dataset_from_directory(
        validation_dir,
        labels='inferred',
        label_mode='int',
        batch_size=opts["batch_size"],
        image_size=(opts["img_height"], opts["img_width"]),
    )


    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = ks.models.Sequential([
        ks.layers.Rescaling(1./255),
        ks.layers.Conv2D(16, 3, padding='same', activation='relu'),
        ks.layers.MaxPooling2D(),
        ks.layers.Conv2D(32, 3, padding='same', activation='relu'),
        ks.layers.MaxPooling2D(),
        ks.layers.Conv2D(64, 3, padding='same', activation='relu'),
        ks.layers.MaxPooling2D(),
        ks.layers.Dropout(0.2),
        ks.layers.Flatten(),
        ks.layers.Dense(128, activation='relu'),
        ks.layers.Dense(len(class_names), name="outputs")
    ])

    model.compile(optimizer='adam',
                loss=ks.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    keras_model_path = 'keras_save.keras'
    model.save(keras_model_path)

if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(description=("Train tool\n"))

        parser.add_argument(
            "dataset_dir",
            type=str,
            help=("directory to look for train and validation subfolders")
        )

        parser.add_argument(
            "--epoch",
            type=int,
            default=8,
            help=("number of epochs for training (default 8)")
        )

        args = parser.parse_args()


        train_dir = Path(args.dataset_dir) / "train"
        validate_dir = Path(args.dataset_dir) / "validation"
        train(args.epoch, train_dir, validate_dir)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
