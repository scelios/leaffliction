#!/usr/bin/env python3

from Utils import load_model
from typing import Any
from pathlib import Path
import argparse
import os
import tensorflow as tf

import matplotlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # set to '' to disable GPU init
# Reduce TensorFlow logs: 0=all, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot


def evaluate_model(restored_keras_model: Any, ds_test: Any,
                   predict_dataset: Any, class_names: list[str]):

    # replace remote tfds load with loading from local directories
    # test_loss, test_acc = restored_keras_model.evaluate(ds_test)
    predictions = restored_keras_model.predict(predict_dataset)
    # collect true labels and predicted labels in the same order
    y_true = [int(lbl.numpy()) for _, lbl in ds_test.unbatch()]
    y_pred = [int(tf.math.argmax(logits).numpy()) for logits in predictions]

    num_classes = len(class_names)
    print("\nPer-class summary:")
    for idx in range(num_classes):
        total = y_true.count(idx)
        correct = sum(1 for t, p in zip(y_true, y_pred)
                      if t == idx and p == idx)
        wrong = total - correct
        accuracy = (100.0 * correct / total) if total > 0 else 0.0
        print(
            f"{class_names[idx]}: total={total}, true={correct},"
            f" false={wrong}, accuracy={accuracy:.1f}%")


if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser(description=("Open model\n"))

        parser.add_argument(
            "--model_path",
            default="keras_save.keras",
            type=str,
            help=("path to the Keras model file")
        )

        parser.add_argument(
            "--validation_dir",
            default="augmented_directory/validation",
            type=str,
            help=("directory to look for validation subfolder")
        )

        args = parser.parse_args()

        keras_model_path = args.model_path
        validation_dir = Path(args.validation_dir)
        keras_model, ds_test, predict_dataset, class_names = load_model(
            keras_model_path, validation_dir)
        evaluate_model(keras_model, ds_test, predict_dataset, class_names)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
