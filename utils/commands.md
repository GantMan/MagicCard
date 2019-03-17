# Download and resize Labelbox images
python -u labelbox_parser.py --labelbox_json_file ./labelbox-export.json --labelbox_class_names "Card" --output_dir ../RCNNCardData --resize_images

# Split into squares?  WTF is this?
python -u image_splitter.py --input_dir ../RCNNCardData/ --output_dir ../RCNNCardData_split

### Separate 5% test
python -u separate_test_and_augmentation_images.py --labelbox_output_dir ../RCNNCardData --image_splitter_output_dir ../RCNNCardData_split/with_labels_only --output_test_dir ../RCNNCardData_test --output_augmentation_dir ../RCNNCardData_aug_raw --labelbox_class_names "Card"

### Create Augmented Images (no color aug)
python -u augmentation.py --input_dir ../RCNNCardData_aug_raw/ --output_dir ../RCNNCardData_aug_out/ --number_of_augmented_images_per_original 30 --no-augment_colour

### Create 5% validation (and rest train)
python -u separate_train_and_val_images.py --input_dir ../RCNNCardData_aug_out/ --output_dir ../RCNNCardData_dataset --labelbox_class_names "Card"

### Then copy the test into the data folder