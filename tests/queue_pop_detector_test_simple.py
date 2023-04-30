import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("queue_pop_detector.h5")

# Load the screenshot and preprocess it
img = cv2.imread("queue_screenshot.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img / 255.0

# Make a prediction
pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]

# Check if the queue has popped
if pred >= 0.10:
    print("The queue has popped!")
else:
    print("The queue has not popped yet.")
