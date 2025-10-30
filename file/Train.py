#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import Utils as u
import Augmentation as aug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras as ks  # noqa: E402      -- must come after os.environ
import tensorflow as tf  # noqa: E402 -- must come after os.environ


def train(epochs, train_dir: Path, validation_dir: Path, batch_size=32):
    opts = {
        "batch_size": batch_size,
        "img_height": 256,
        "img_width": 256,
    }

    train_ds: Any = ks.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=opts["batch_size"],
        image_size=(opts["img_height"], opts["img_width"]),
    )

    validation_ds: Any = ks.utils.image_dataset_from_directory(
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
                  loss=ks.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
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


def is_dir_balanced(dir_path: Path, num_validation: int) -> bool:
    # check if each subdirectory has around the same number of images (+- 5%)
    # as the other or at least num_validation images
    from collections import defaultdict
    grouped = defaultdict(int)
    for img_path in u.get_all_images(dir_path):
        variation = img_path.parts[-2]
        grouped[variation] += 1
    counts = list(grouped.values())
    min_count = min(counts)
    max_count = max(counts)
    if min_count < num_validation:
        return False
    if max_count > min_count * 1.05:
        return False
    return True


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

        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help=("batch size to use during training"
                  "(usefull for memory constrained systems)")
        )

        args = parser.parse_args()

        # Check if dataset is balanced, if not augment it
        if is_dir_balanced(Path(args.dataset_dir), num_validation=50) is False:
            print("Dataset is not balanced, generating augmentd_directory")
            image_dir = str(Path(args.dataset_dir))
            args.dataset_dir = str(Path(args.dataset_dir) /
                                   "augmented_directory")
            aug.main(image_dir, args.dataset_dir, 16)

        print(f"Using dataset directory: {args.dataset_dir}")
        train_dir = Path(args.dataset_dir) / "train"
        validate_dir = Path(args.dataset_dir) / "validation"
        train(args.epoch, train_dir, validate_dir, args.batch_size)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
