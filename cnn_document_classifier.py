import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 128
img_width = 128
batch_size = 32

train_data = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_data.flow_from_directory(
    "dataset/train",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = train_data.flow_from_directory(
    "dataset/train",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_gen.num_classes

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

test_data = ImageDataGenerator(rescale=1./255)
test_gen = test_data.flow_from_directory(
    "dataset/test",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

loss, accuracy = model.evaluate(test_gen)
print("Test accuracy:", round(accuracy, 4))

model.save("document_layout_cnn.h5")
