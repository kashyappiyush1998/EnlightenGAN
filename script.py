import os
import argparse
from predict import get_enlightened_image

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

# image = get_enlightened_image()
# print(image)
# image.save('new_image.png')