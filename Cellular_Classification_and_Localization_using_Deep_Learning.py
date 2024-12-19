
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Parameters
img_height, img_width = 224, 224
batch_size = 32
num_classes = 2  # Modify based on dataset

# Dataset Paths
data_dir = "./biological_images"
labels_file = "./labels.csv"  # CSV file mapping image names to labels

# Load and Prepare Dataset
data = pd.read_csv(labels_file)
data["image_path"] = data_dir + "/" + data["image_name"]

# Verify Dataset
if "label" not in data.columns or "image_path" not in data.columns:
    raise ValueError("Dataset must contain 'label' and 'image_path' columns.")

# Splitting Dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col="image_path",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow_from_dataframe(
    test_data,
    x_col="image_path",
    y_col="label",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# Build Model
def build_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False  # Freeze the base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

model = build_model()

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    verbose=1
)

# Save Model
model.save("cellular_classification_model.h5")

# Evaluate Model
val_labels = test_data['label'].values
val_predictions = model.predict(val_generator, verbose=1)
val_predicted_classes = np.argmax(val_predictions, axis=1)

f1 = f1_score(val_labels, val_predicted_classes, average='weighted')
precision = precision_score(val_labels, val_predicted_classes, average='weighted')
recall = recall_score(val_labels, val_predicted_classes, average='weighted')
conf_matrix = confusion_matrix(val_labels, val_predicted_classes)

print("Classification Report:\n", classification_report(val_labels, val_predicted_classes))
print("Confusion Matrix:\n", conf_matrix)
print(f"F1 Score: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Visualize Training Performance
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("training_accuracy.png")
plt.show()

# Grad-CAM Implementation
def grad_cam(input_model, image, layer_name, class_index):
    grad_model = Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()[0]
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return cv2.resize(heatmap, (img_width, img_height))

# Test Grad-CAM on a Sample Image
sample_image_path = test_data.iloc[0]['image_path']
sample_image = cv2.imread(sample_image_path)
sample_image = cv2.resize(sample_image, (img_height, img_width)) / 255.0

predicted_class = np.argmax(model.predict(np.array([sample_image])))
heatmap = grad_cam(model, sample_image, 'top_conv', predicted_class)

# Overlay Heatmap
overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
final_image = cv2.addWeighted(np.uint8(sample_image * 255), 0.7, overlay, 0.3, 0)

# Display Heatmap
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(sample_image)

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(final_image)
plt.savefig("grad_cam_output.png")
plt.show()
