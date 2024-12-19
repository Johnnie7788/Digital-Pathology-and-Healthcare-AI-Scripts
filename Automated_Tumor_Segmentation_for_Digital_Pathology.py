#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Automated Tumor Segmentation for Digital Pathology

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.metrics import MeanIoU
import matplotlib.pyplot as plt
import cv2
import os

# Parameters
img_height, img_width = 128, 128
batch_size = 16
log_dir = "logs/fit/"

# Custom Data Generator to Load Images and Masks
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, img_height, img_width, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.indices = np.arange(len(self.image_paths))
        np.random.shuffle(self.indices)  # Shuffle the data
        self.augment = augment

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        images = [self._load_image(self.image_paths[i]) for i in batch_indices]
        masks = [self._load_mask(self.mask_paths[i]) for i in batch_indices]
        if self.augment:
            images, masks = self._augment(images, masks)
        return np.array(images), np.array(masks)

    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found at path: {path}")
        image = cv2.resize(image, (self.img_width, self.img_height))
        image = image / 255.0  # Normalize
        return np.expand_dims(image, axis=-1)

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found at path: {path}")
        mask = cv2.resize(mask, (self.img_width, self.img_height))
        mask = mask / 255.0  # Ensure binary mask
        mask = np.round(mask)  # Ensure values are 0 or 1
        return np.expand_dims(mask, axis=-1)

    def _augment(self, images, masks):
        augmented_images, augmented_masks = [], []
        for img, mask in zip(images, masks):
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)
            augmented_images.append(img)
            augmented_masks.append(mask)
        return augmented_images, augmented_masks

# Load Image and Mask Paths
def get_file_paths(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")])

image_paths = get_file_paths('./segmentation_data/images')
mask_paths = get_file_paths('./segmentation_data/masks')

# Verify Image-Mask Matching
if len(image_paths) != len(mask_paths):
    raise ValueError("Number of images and masks do not match!")

# Split into training and validation
split = int(0.8 * len(image_paths))
train_image_paths, val_image_paths = image_paths[:split], image_paths[split:]
train_mask_paths, val_mask_paths = mask_paths[:split], mask_paths[split:]

# Create Data Generators
train_generator = CustomDataGenerator(train_image_paths, train_mask_paths, batch_size, img_height, img_width, augment=True)
val_generator = CustomDataGenerator(val_image_paths, val_mask_paths, batch_size, img_height, img_width, augment=False)

# U-Net Model
inputs = Input((img_height, img_width, 1))

# Encoder
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
conv1 = BatchNormalization()(conv1)
conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Dropout(0.3)(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Dropout(0.3)(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Dropout(0.4)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

# Bottleneck
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Dropout(0.4)(conv4)

# Decoder
up5 = UpSampling2D(size=(2, 2))(conv4)
up5 = concatenate([up5, conv3], axis=-1)
conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(up5)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Dropout(0.4)(conv5)

up6 = UpSampling2D(size=(2, 2))(conv5)
up6 = concatenate([up6, conv2], axis=-1)
conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Dropout(0.3)(conv6)

up7 = UpSampling2D(size=(2, 2))(conv6)
up7 = concatenate([up7, conv1], axis=-1)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Dropout(0.3)(conv7)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

model = Model(inputs=[inputs], outputs=[outputs])

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=lambda y_true, y_pred: 1 - tf.reduce_mean((2 * tf.reduce_sum(y_true * y_pred) + 1e-7) /
                                                        (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)),
              metrics=[MeanIoU(num_classes=2)])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("unet_tumor_segmentation_best_model.h5", save_best_only=True),
    CSVLogger("training_log.csv"),
    TensorBoard(log_dir=log_dir)
]

# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    epochs=50,
    callbacks=callbacks
)

# Save the Final Model
model.save("unet_tumor_segmentation_model.h5")

# Visualization Example
sample_image = train_generator[0][0][0]  # First image in the batch
input_image = np.expand_dims(sample_image, axis=0)
predicted_mask = model.predict(input_image)[0, :, :, 0]

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(sample_image.squeeze(), cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.imshow(predicted_mask, cmap='jet', alpha=0.5)
plt.show()

