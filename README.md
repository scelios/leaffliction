# leaffliction

## links

- TensorFlow image classification tutorial: https://www.tensorflow.org/tutorials/images/classification
- TS image exmaple https://www.tensorflow.org/tutorials/load_data/images
- Wikipedia CNN explanation: https://en.wikipedia.org/wiki/Convolutional_neural_network
- PlantCV: https://plantcv.readthedocs.io/en/stable/

# CNN

> **Convolutional Neural Network**

This project aim to create an AI wich will determine if a leaf is healthy or not and wich affliction it has.
It will also determine wich type of leaf it is.

We have a dockerfile that can be launch through the makefile with ``make``
Or you could use uv with ``uv sync`` to get the environment

Augmentation file will artificially augment the size of th dataset by flipping, rotating, blur, shear, etc
It will then create 2 folder named train and validation where the dataset will be balanced

Distribution file will show the distribution of the subdirectories images

Transformation will show through pictures how the ia can analyze the pictures

Train will train the model with keras

Predict will predict the class of an image or a folder with a confusion matrix

---

## Overview

leaffliction is a small image-classification pipeline focused on detecting leaf type and common afflictions (apple rust, scab, grape spots, etc.). The repository provides tools to:

- prepare and augment datasets ([file/Augmentation.py](file/Augmentation.py))
- visualize dataset distribution ([file/Distribution.py](file/Distribution.py))
- transform and analyze single images with PlantCV ([file/Transformation.py](file/Transformation.py))
- train a simple CNN ([file/Train.py](file/Train.py))
- evaluate and predict using a saved Keras model ([file/Evaluate.py](file/Evaluate.py); [file/Predict.py](file/Predict.py))
- utilities for image IO, parallel processing and dataset loading ([file/Utils.py](file/Utils.py), [`Utils.load_image`](file/Utils.py), [`Utils.save_image`](file/Utils.py), [`Utils.get_all_images`](file/Utils.py), [`Utils.load_model`](file/Utils.py))

---

## Quickstart

```
# Usage:
# make ARGS=some_image.JPG

# Or without make
# uv sync
# source .venv/bin/activate
# python file/Train.py file/images
# python file/Predict.py someimage.JPG
```

---

## Typical workflow

1. Inspect dataset distribution
   - Run the distribution helper to see imbalance:
     ```
     python file/Distribution.py file/images
     ```
     This uses [`Distribution.create_charts`](file/Distribution.py).

2. Create balanced training/validation sets and augment
   - Use [file/Augmentation.py](file/Augmentation.py). It relies on helpers in [file/Utils.py](file/Utils.py), e.g. [`Utils.get_all_images`](file/Utils.py) and [`Utils.parallel_process`](file/Utils.py).
   - Example (directory -> `augmented_directory`):
     ```
     python file/Augmentation.py file/images --output augmented_directory --validation 16
     ```

3. Inspect or transform specific images
   - Use [file/Transformation.py](file/Transformation.py) to run plantCV analyses and show multiple transformation plots:
     ```
     python file/Transformation.py file/images/Apple_healthy/example.jpg
     ```
     (uses [`Transformation.main`](file/Transformation.py))

4. Train the CNN
   - Use [file/Train.py](file/Train.py) [`Train.train`](file/Train.py). It expects a directory with `train/` and `validation/` subfolders (the `augmented_directory` layout matches this).
   - Example:
     ```
     python file/Train.py augmented_directory --epoch 8
     ```
     This will save the trained model to `keras_save.keras`.

5. Evaluate and predict
   - Load the saved model and evaluate over validation:
     ```
     python file/Evaluate.py --model_path keras_save.keras --validation_dir augmented_directory/validation
     ```
     (uses [`Utils.load_model`](file/Utils.py) to build datasets and load the model; evaluation logic lives in [`Evaluate.evaluate_model`](file/Evaluate.py))

   - Single-image prediction:
     ```
     python file/Predict.py path/to/image.jpg --model_path keras_save.keras --validation_dir augmented_directory/validation
     ```
     (calls [`Predict.predict`](file/Predict.py))
