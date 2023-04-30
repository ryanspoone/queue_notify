import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.applications import VGG16

# Set GPU memory growth to limit GPU usage to 80%
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=0.8 * 1024
                    )
                ],
            )
    except RuntimeError as e:
        print(e)

# Define the data augmentation parameters
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Use the ImageDataGenerator to create train and validation generators
train_generator = train_datagen.flow_from_directory(
    "train",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
)

val_generator = val_datagen.flow_from_directory(
    "val",
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
)

# Load a pre-trained model for transfer learning
pretrained_model = VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Freeze the pre-trained layers
for layer in pretrained_model.layers:
    layer.trainable = False

# Create a new model with the pre-trained layers and some new layers
model = Sequential(
    [
        pretrained_model,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"]
)


# Define the path where the model weights will be saved
checkpoint_filepath = "queue_pop_detector.h5"

# Create a ModelCheckpoint callback to save the model weights after each epoch
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_best_only=False,
    monitor="val_loss",
)

# Train the model using the generators and the ModelCheckpoint callback
try:
    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=5, mode="max", verbose=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=100,
        validation_data=val_generator,
        validation_steps=val_generator.n // val_generator.batch_size,
        callbacks=[checkpoint_callback, early_stop],
    )
except KeyboardInterrupt:
    print("Training interrupted")

# Save the model
model.save("queue_pop_detector.h5")
