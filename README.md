# leaffliction

## links

- https://www.tensorflow.org/tutorials/images/classification
- https://www.tensorflow.org/tutorials/load_data/images

## TODO

> 🏁 **Finish for this week** `31/10/2025`

- [ ] make plant CV work to extract the leaf shape fom the backgroud
    - half day
- [ ] _maybe_ Modify augmentation to use `tf.keras` for image split & dataset augementation,
    - half day, if time
- [x] split dataset before into validation & train (pre augment)
    - half half day
- [ ] use `tensorflow` to setup CNN to learn leaf categories, aplle_rust, grape_healthy, etc ...
    - 2 days
- [ ] eval project
    - half day
- [ ] learn about CNN and the training process, enough to explain for eval
    - half day (?)
- [x] get distracted
    - sporatic, 1 day
- [ ] play katan
    - 4 hours
- [ ] do transformation for a directory, save plot for each image

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

Core dev files and metadata:
- [dockerfile](dockerfile) — base image and Python deps install
- [docker-compose.yaml](docker-compose.yaml) — service to run container
- [Makefile](Makefile) — convenience targets (build, exec, images)
- [pyproject.toml](pyproject.toml), [file/requirements.txt](file/requirements.txt)

---

## Quickstart

1. Get the image through the makefile
   - Using Makefile:
     ```
     make images
     ```
2. Build and open the development container:
   - Using Makefile:
     ```
     make
     ```
     (runs the `docker compose up --build -d && docker exec -it python /bin/bash` flow declared in [Makefile](Makefile))
   - Or build manually with Docker:
     - Build: docker build -t leaffliction -f dockerfile .
     - Run: docker compose up --build -d (uses docker-compose.yaml)

3. Inside the container the working directory is mounted to `/usr/app` and contains the `file/` scripts. Use them from there.

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

---

## File matrix (quick links)

- Data & images
  - file/images/ (raw images)
  - file/augmented_directory/ (generated train/validation sets)

- Scripts
  - [file/Augmentation.py](file/Augmentation.py) — augmentation helpers and CLI (`Augmentation.process_one_image`, `Augmentation.main`)
  - [file/Transformation.py](file/Transformation.py) — PlantCV visualizations (`Transformation.main`)
  - [file/Distribution.py](file/Distribution.py) — dataset distribution plotter (`Distribution.create_charts`)
  - [file/Train.py](file/Train.py) — training script and model save (`Train.train`)
  - [file/Evaluate.py](file/Evaluate.py) — evaluation helper (`Evaluate.evaluate_model`)
  - [file/Predict.py](file/Predict.py) — single-image prediction (`Predict.predict`)
  - [file/Utils.py](file/Utils.py) — IO, parallel, dataset loader (`Utils.load_image`, `Utils.save_image`, `Utils.get_all_images`, `Utils.load_model`)

- Dev / infra
  - [dockerfile](dockerfile)
  - [docker-compose.yaml](docker-compose.yaml)
  - [Makefile](Makefile)
  - [pyproject.toml](pyproject.toml)
  - [file/requirements.txt](file/requirements.txt)

---

## Notes & tips

- The project uses Matplotlib with the TkAgg backend in multiple scripts (see [file/Utils.py](file/Utils.py), [file/Distribution.py](file/Distribution.py), [file/Transformation.py](file/Transformation.py)). When running inside Docker ensure X11 forwarding is configured (the docker-compose mounts /tmp/.X11-unix and passes DISPLAY).
- When datasets are large, use the augmentation pipeline in [file/Augmentation.py](file/Augmentation.py) together with [`Utils.parallel_process`](file/Utils.py) to speed up I/O-bound work.
- Models are saved to `keras_save.keras` by default by [file/Train.py](file/Train.py). Use [file/Utils.py](file/Utils.py) [`Utils.load_model`](file/Utils.py) to load the model and rebuild tf.data datasets from the local `train/` and `validation/` folders.

---

## Acknowledgements

- TensorFlow image classification tutorial: https://www.tensorflow.org/tutorials/images/classification
- Wikipedia CNN explanation: https://en.wikipedia.org/wiki/Convolutional_neural_network
- PlantCV: https://plantcv.readthedocs.io/en/stable/

---

**TODO write the breakdown of how it all supposed to work**


## some shell commands

Transform the unit test files to the directory structure that is required.

```bash
ls -1 | xargs -I{} bash -c "echo {} | sed 's/[0-9]*\\.JPG$//'" | xargs -I{} bash -c "mkdir -p {} & mv {}*.JPG {}"
```     