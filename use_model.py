import cv2
import mrcnn.config
import mrcnn.visualize
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


# Configuration that will be used by the Mask-RCNN library
class ObjectDetectorConfig(mrcnn.config.Config):
    NAME = "card"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1  # custom object + background class


# Root directory of the project
ROOT_DIR = Path(".")
MODEL_DIR = ROOT_DIR / "training_logs"

# Local path to trained weights file (make sure you update this)
TRAINED_MODEL_PATH = MODEL_DIR / "card20190315T1656" / "mask_rcnn_card_0030.h5"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=ObjectDetectorConfig())

# Load pre-trained model
model.load_weights(str(TRAINED_MODEL_PATH), by_name=True)

# COCO Class names
class_names = ['BG', 'blank_card']

# Load the image we want to run detection on
# image_path = "../RCNNCardData_dataset/stage1_test/IMG_5624_inst_0_class_Card_seg_0.png/images/IMG_5624_inst_0_class_Card_seg_0.png"
# image_path = "../RCNNCardData_dataset/stage1_test/MVIMG_20190310_152531_inst_0_class_Card_seg_0.png/images/MVIMG_20190310_152531_inst_0_class_Card_seg_0.png"
image_path = "../RCNNCardData_dataset/stage1_test/Photo on 3-10-19 at 3.17 PM_inst_0_class_Card_seg_0.png/images/Photo on 3-10-19 at 3.17 PM_inst_0_class_Card_seg_0.png"
image = cv2.imread(image_path)

# Convert the image from BGR color (which OpenCV uses) to RGB color
rgb_image = image[:, :, ::-1]

# Run the image through the model
results = model.detect([rgb_image], verbose=1)

# Visualize results
r = results[0]
mrcnn.visualize.display_instances(rgb_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
