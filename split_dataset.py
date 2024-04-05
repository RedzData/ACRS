from sklearn.model_selection import train_test_split
import os

# Directory containing the images
base_dir = "C:/Users/redzo/ACRS/Planeimages"

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
                    # Append the image filename and label
                    image_filenames.append(os.path.join(model_dir, filename))
                    labels.append(f"{aircraft_type}_{model}")

# Split the dataset into training, validation, and test sets
train_images, temp_images, train_labels, temp_labels = train_test_split(image_filenames, labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# Print the sizes of each set
print(f"Training set size: {len(train_images)}")
print(f"Validation set size: {len(val_images)}")
print(f"Test set size: {len(test_images)}")

# Now you have your datasets ready for training, validation, and testing
# train_images, train_labels --> Training set
# val_images, val_labels --> Validation set
# test_images, test_labels --> Test set
