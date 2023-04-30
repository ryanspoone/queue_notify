import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import shutil

# Load the trained model
model = load_model("queue_pop_detector.h5")

# Create directory to store incorrectly classified images
if not os.path.exists("incorrectly_classified"):
    os.makedirs("incorrectly_classified")

# Loop through each image in the train and val sets
for dataset in ["train", "val"]:
    for label in ["not_queue_pop", "queue_pop"]:
        data_dir = os.path.join(dataset, label)
        for img_file in os.listdir(data_dir):
            # Load the image and preprocess it
            img_path = os.path.join(data_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0

            # Make a prediction
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
            is_queue_pop = pred >= 0.95

            # Check if the prediction is correct
            if label == "queue_pop" and not is_queue_pop:
                shutil.move(
                    img_path, os.path.join("incorrectly_classified", img_file)
                )
            elif label == "not_queue_pop" and is_queue_pop:
                shutil.move(
                    img_path, os.path.join("incorrectly_classified", img_file)
                )
