#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import tempfile
import shutil
import Transformation as tfm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras as ks  # noqa: E402      -- must come after os.environ
import tensorflow as tf  # noqa: E402 -- must come after os.environ


def predict(model, predict_image_path, class_names):
    img = ks.utils.load_img(
        predict_image_path, target_size=(256, 256)
    )
    img_array = ks.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions.argmax(axis=-1)[0]

    print("===       DL classification       ===")
    print(f"Class predicted: \033[32m{class_names[score]}\033[0m")

    mask = tfm.mask(img_array[0].numpy().astype(np.uint8))
    # ensure binary 0/255 mask (plantcv usually returns this, but be safe)
    mask_bin = (mask > 0).astype(np.uint8) * 255
    # remove background from original image using the mask
    orig = img_array[0].numpy().astype(np.uint8)
    masked = tfm.pcv.apply_mask(img=orig, mask=mask_bin, mask_color='white')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(masked)
    plt.axis("off")
    plt.title(f"Predicted class: {class_names[score]}")
    plt.show()


def evaluate_model(model, data_dir):
    infer_ds = ks.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=(256, 256),
        shuffle=False
    )

    # Get ground truth labels
    y_true = np.concatenate([y.numpy() for x, y in infer_ds], axis=0)

    # Predict
    predictions = model.predict(infer_ds)
    y_pred = tf.argmax(predictions, axis=1).numpy()

    print("y_true", y_true)
    print("y_pred", y_pred)

    # Confusion matrix
    cm = skm.confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    print("CM", cm)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    print("cm_norm", cm_norm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = skm.ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format=".2f", colorbar=True)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def load_model(keras_model_path, validation_dir):
    validation_dir = validation_dir
    opts = {
        "batch_size": 32,
        "img_height": 256,
        "img_width": 256,
    }
    # replace remote tfds load with loading from local directories
    ds_train: Any = ks.utils.image_dataset_from_directory(
        validation_dir,
        labels='inferred',
        label_mode='int',
        batch_size=opts["batch_size"],
        image_size=(opts["img_height"], opts["img_width"]),
        shuffle=True,
    )

    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # derive class names from the directory dataset
    class_names = ds_train.class_names

    # load the model back
    restored_keras_model = ks.models.load_model(keras_model_path)
    return restored_keras_model, class_names


def gen_class_dir(predict_path: Path, class_names: list):
    if all((predict_path / cls).is_dir() for cls in class_names):
        return predict_path, False

    # Create temporary copy of the directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="leaffliction_classdir_"))
    shutil.copytree(predict_path, tmp_dir, dirs_exist_ok=True)

    # Add any missing class folders
    for cls in class_names:
        (tmp_dir / cls).mkdir(parents=True, exist_ok=True)

    return tmp_dir, True


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=("Train tool\n"))

        parser.add_argument(
            "path",
            type=str,
            help=(
                "predict the class of path for "
                "either a single image or a direcory for stats")
        )

        parser.add_argument(
            "--model_path",
            type=str,
            default="leaffliction.keras",
            help=("model file, should be a .keras file")
        )

        parser.add_argument(
            "--validation_dir",
            type=str,
            default="augmented_directory/validation",
            help=("directory to look for validation subfolder")
        )

        args = parser.parse_args()

        keras_model_path = Path(args.model_path)
        validation_dir = Path(args.validation_dir)
        keras_model, class_names = load_model(keras_model_path, validation_dir)

        if not keras_model:
            raise FileNotFoundError("Could not load model")

        # model = ks.saving.load_model(Path(args.model_path))
        # if not model:
            # raise FileNotFoundError("Could not load model")

        ks.models.load_model

        predict_path = Path(args.path)

        if predict_path.is_dir():
            fixed_pred_path, is_tmp_dir = gen_class_dir(
                predict_path, class_names)
            if is_tmp_dir:
                print("tmp path is", fixed_pred_path)
            evaluate_model(keras_model, fixed_pred_path)
            if (is_tmp_dir):
                shutil.rmtree(fixed_pred_path)
        else:
            predict(keras_model, predict_path, class_names)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
