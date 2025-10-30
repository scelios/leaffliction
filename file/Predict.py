#!/usr/bin/env python3

import argparse
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
from Utils import load_model


def predict(model_path, predict_image_path, class_names=None):
    img = tf.keras.utils.load_img(
        predict_image_path, target_size=(256, 256)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model_path.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        f"This image most likely belongs to class "
        f"{class_names[tf.argmax(score)]} with a {100 * tf.reduce_max(score):.2f} percent confidence."
    )


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(description=("Train tool\n"))

        parser.add_argument(
            "predict_image_path",
            type=str,
            help=("prediction")
        )

        parser.add_argument(
            "--model_path",
            type=str,
            default="keras_save.keras",
            help=("model file, usually a .keras file")
        )

        parser.add_argument(
            "--validation_dir",
            type=str,
            default="augmented_directory/validation",
            help=("directory to look for validation subfolder")
        )

        args = parser.parse_args()
        keras_model_path = args.model_path
        validation_dir = Path(args.validation_dir)
        keras_model, ds_test, predict_dataset, class_names = load_model(keras_model_path, validation_dir)
        predict(keras_model, Path(args.predict_image_path), class_names)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)