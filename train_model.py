# Roughly based on the balloon.py demo included with Matterport's Mask R-CNN implementation which
# is licensed under the MIT License (see LICENSE for details).
# This is based on code originally written by Waleed Abdulla but re-written to support data annotated by Labelbox.

import os
import warnings
from pathlib import Path
import xmltodict
import json
import numpy as np
import skimage.draw
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = Path("../")
DATASET_DIR = "RCNNCardData_dataset"
COCO_WEIGHTS_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

# Download COCO trained weights if you don't already have them.
if not COCO_WEIGHTS_PATH.exists():
    utils.download_trained_weights(str(COCO_WEIGHTS_PATH))

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = Path(".") / "training_logs"

# Where the training images and annotation files live
DATASET_PATH = ROOT_DIR / DATASET_DIR

# Start from the pre-trained COCO model. Change this path if you want to pick up training from a prior
# checkpoint file in your ./training_logs folder.
WEIGHTS_TO_START_FROM = COCO_WEIGHTS_PATH # DEFAULT_LOGS_DIR / "card_30.h5"

# Object Name
CUSTOM_NAME = "card"

class ObjectDetectorConfig(Config):
    NAME = CUSTOM_NAME
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + your custom object
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


class RectLabelDataset(utils.Dataset):

    def load_training_images(self, dataset_dir, subset):
        dataset_dir = dataset_dir / subset

        # Add classes. We have only one class to add since this model only detects one kind of object.
        self.add_class(CUSTOM_NAME, 1, CUSTOM_NAME)

        # get folders with PNG extention (skip evil hidden files like .DS_store)
        for data_item in dataset_dir.glob("*.png"):
            data_name = data_item.parts[-1] # filename
            image_path = os.path.join(data_item, "images", data_name)
            mask_path = os.path.join(data_item, "masks", data_name)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source=CUSTOM_NAME,
                image_id=data_name,
                path=image_path,
                width=width, height=height,
                mask_path=mask_path
                # something here objects or polygons
            )

    def load_mask(self, image_id):
        # Look up the current image id
        info = self.image_info[image_id]
        # Get image and add single dimension
        mask_image = np.expand_dims(skimage.io.imread(info["mask_path"]), axis=-1)
        return mask_image.astype(np.bool), np.ones([mask_image.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        # Get the path for the image
        info = self.image_info[image_id]
        return info["path"]


def train(model):
    # Load the training data set
    dataset_train = RectLabelDataset()
    dataset_train.load_training_images(DATASET_PATH, "stage1_train")
    dataset_train.prepare()

    # Load the validation data set
    dataset_val = RectLabelDataset()
    dataset_val.load_training_images(DATASET_PATH, "val")
    dataset_val.prepare()

    with warnings.catch_warnings():
        # Suppress annoying skimage warning due to code inside Mask R-CNN.
        # Not needed, but makes the output easier to read until Mask R-CNN is updated.
        warnings.simplefilter("ignore")

        # Re-train the model on a small data set. If you are training from scratch with a huge data set,
        # you'd want to train longer and customize these settings.
        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads'
        )


# Load config
config = ObjectDetectorConfig()
config.display()

# Create the model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# Load the weights we are going to start with
# Note: If you are picking up from a training checkpoint instead of the COCO weights, remove the excluded layers.
model.load_weights(str(WEIGHTS_TO_START_FROM), by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Run the training process
train(model)
