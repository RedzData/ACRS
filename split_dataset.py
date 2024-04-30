import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Directory containing the images
base_dir = "C:/Users/redzo/ACRS/Planeimages"

# Target dimensions for resizing
image_height = 224
image_width = 224

# List of image filenames and their corresponding labels
image_filenames = []
labels = []

# Iterate over each aircraft type (Boeing and Airbus)
for aircraft_type in ["Boeing", "Airbus"]:
    # Iterate over each model within the aircraft type
    for model in os.listdir(os.path.join(base_dir, aircraft_type)):
        model_dir = os.path.join(base_dir, aircraft_type, model)
        # Check if it's a directory
        if os.path.isdir(model_dir):
            # Iterate over each image file in the model directory
            for filename in os.listdir(model_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Read the image
                    image_path = os.path.join(model_dir, filename)
                    image = cv2.imread(image_path)
                    # Resize the image
                    resized_image = cv2.resize(image, (image_width, image_height))
                    # Append the resized image and label
                    image_filenames.append(resized_image)
                    labels.append(1 if aircraft_type == "Boeing" else 0)  # Encode labels (1 for Boeing, 0 for Airbus)

# Convert lists to numpy arrays
X = np.array(image_filenames)
y = np.array(labels)

# Split the dataset into training, validation, and test sets
train_images, temp_images, train_labels, temp_labels = train_test_split(X, y, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# Define model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    epochs=10,  # You can adjust the number of epochs as needed
                    batch_size=32,
                    validation_data=(val_images, val_labels))

# Evaluate on test set
loss, accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", accuracy)
