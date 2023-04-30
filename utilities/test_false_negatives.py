import os
import cv2
import numpy as np
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the directory path of the current script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = load_model(os.path.join(script_dir, "queue_pop_detector.h5"))


# Define a function to test for false negatives
def test_false_negatives(directory, verbose=False):
    print(directory)

    # Create an ImageDataGenerator for loading the images from disk
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load the images in batches using flow_from_directory
    generator = datagen.flow_from_directory(
        directory,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False,
        interpolation="nearest",
        color_mode="rgb",
    )

    # Load all the images in the directory into a numpy array
    images = []
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            images.append(img)
    images = np.array(images)

    # Predict the labels for each image in the numpy array
    preds = model.predict(
        datagen.flow(images, batch_size=32, shuffle=False), verbose=0
    )

    # Loop over the predictions and filenames and print the filenames of any false negatives
    for pred, filename in zip(preds, generator.filenames):
        if pred > 0.5:
            if verbose:
                print(f"{filename} is a false negative")
            else:
                print(filename)
                shutil.move(
                    os.path.join(directory, filename),
                    os.path.join(script_dir, "false_negatives"),
                )


# Test for false negatives in the train directory
train_dir = os.path.join(script_dir, "train", "not_queue_pop")
test_false_negatives(train_dir, verbose=False)

# Test for false negatives in the val directory
val_dir = os.path.join(script_dir, "val", "not_queue_pop")
test_false_negatives(val_dir, verbose=False)
