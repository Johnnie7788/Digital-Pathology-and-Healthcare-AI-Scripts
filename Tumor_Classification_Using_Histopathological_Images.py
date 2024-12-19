

# Tumor Classification Using Histopathological Images


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import cv2

# Load and preprocess data (example paths, modify as needed)
data_dir = "./histopathological_images"
img_height, img_width = 224, 224
batch_size = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Build the CNN model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(img_height, img_width, 3)))
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
out = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,
    verbose=1
)

# Save the model
model.save("multiclass_tumor_classifier_model.h5")

# Grad-CAM implementation with tumor marker
def grad_cam(input_model, image, layer_name, class_index):
    grad_model = Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[0][class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    casted_grads = tf.cast(grads > 0, "float32") * grads
    weights = tf.reduce_mean(casted_grads, axis=(0, 1))

    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1).numpy()
    cam = cv2.resize(cam, (img_width, img_height))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    return heatmap

# Automatic tumor detection and marking
def detect_and_mark_tumor(image, heatmap, threshold=0.5):
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_map = cv2.threshold(heatmap, int(threshold * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marked_image = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(marked_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return marked_image

# Visualizing Grad-CAM and tumor marker
sample_image, sample_label = train_generator.next()
sample_image = sample_image[0]
sample_class = np.argmax(sample_label[0])

heatmap = grad_cam(model, sample_image, 'block5_conv3', sample_class)
marked_image = detect_and_mark_tumor(sample_image * 255, heatmap)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Grad-CAM Visualization")
plt.imshow(sample_image)
plt.imshow(heatmap, cmap='jet', alpha=0.5)

plt.subplot(1, 2, 2)
plt.title("Tumor Detection with Marker")
plt.imshow(marked_image.astype(np.uint8))
plt.show()

# Automatic recommendation system
def generate_recommendation(predictions):
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]
    recommendations = {
        0: "Low likelihood of tumor. Continue routine monitoring.",
        1: "Benign tumor detected. Recommend periodic check-ups.",
        2: "High likelihood of malignant tumor detected. Recommend immediate medical evaluation.",
    }
    return recommendations.get(class_index, "Unknown condition. Consult a specialist."), confidence

# Test recommendation
predictions = model.predict(np.array([sample_image]))[0]
recommendation, confidence = generate_recommendation(predictions)
print(f"Predicted Class: {np.argmax(predictions)}, Confidence: {confidence:.2f}, Recommendation: {recommendation}")

