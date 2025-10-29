

import argparse
from pathlib import Path

def predict(model_path, predict_image_path):
    print("MODEL PATH", model_path)
    print("PREDICT IMAGE PATH", predict_image_path)
    pass


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

        args = parser.parse_args()

        predict(Path(args.model_path), Path(args.predict_image_path))

    except Exception as e:
        print(f"Error: {e}")
        exit(1)