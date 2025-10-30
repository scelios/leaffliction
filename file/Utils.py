#!/usr/bin/env python3

from ast import List
import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot

import skimage as ski
from skimage import io, exposure
from skimage.util import img_as_ubyte
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import tensorflow as tf


def load_image(path):
    """Load an image as float from path"""
    return ski.util.img_as_float(ski.io.imread(path))


def save_image(image, save_path: Path):
    """Save image of any type, (converted with img_as_ubyte)"""
    save_path.parent.mkdir(parents=True,  exist_ok=True)
    image = ski.util.img_as_ubyte(image)
    ski.io.imsave(save_path, image, check_contrast=False)


def gen_path(image_path: Path, filename_suffix: str, out_dir="transformed", top_folder="train") -> Path:
    """
    Generate a new output path by merging the given image_path into out_dir.
    Keeps the last two directory levels of image_path, and replaces the filename
    with filename_suffix.
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)

    parts = image_path.parts
    # Get up to last 3 parts (2 dirs + filename)
    sub_parts = parts[-3:] if len(parts) >= 3 else parts
    # Replace filename with the provided suffix
    if sub_parts:
        sub_parts = list(sub_parts)
        sub_parts[-1] = filename_suffix
        sub_parts[0] = top_folder

    new_subpath = Path(*sub_parts)
    return out_dir / new_subpath


def parallel_process(items, func, n_jobs=-1, use_tqdm=True):
    """Launch jobs in parallel with a tqdm progress bar"""
    if use_tqdm:
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(func)(item) for item in tqdm(items)
        )
    else:
        return Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in items)


def get_all_images(root_dir, exts=(".jpg", ".jpg", ".jpeg", ".png", ".tif", ".bmp")):
    root_dir = Path(root_dir)
    # Use rglob for recursive search; match extensions case-insensitively
    image_paths = [
        p for p in root_dir.rglob("*")
        if p.suffix.lower() in exts
    ]
    return sorted(image_paths)

def load_model(keras_model_path: str, validation_dir: Path):
    train_dir = validation_dir.parent / "train/"
    validation_dir = validation_dir
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

    # predict on the entire validation set (preserves batching)
    predict_dataset = ds_test.map(lambda x, y: x)

    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # derive class names from the directory dataset
    class_names = ds_train.class_names
    print("Class names:", class_names)

    # load the model back
    restored_keras_model = tf.keras.models.load_model(keras_model_path)
    restored_keras_model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return restored_keras_model, ds_test, predict_dataset, class_names
