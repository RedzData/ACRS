import os
import cv2

# Original script for resizing Boeing images
# Directory containing the images
boeing_dir = "C:/Users/redzo/ACRS/Planeimages/Boeing/B737-100"

# Target dimensions
image_height = 1079
image_width = 1600

# Iterate over each image file in the Boeing directory
for filename in os.listdir(boeing_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(boeing_dir, filename)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, (image_width, image_height))

        # Save the resized image
        cv2.imwrite(image_path, resized_image)

print("Boeing images resized successfully.")

# Script for resizing Airbus images
# Directory containing the images
airbus_dir = "C:/Users/redzo/ACRS/Planeimages/Airbus/A-300"

# Iterate over each image file in the Airbus directory
for filename in os.listdir(airbus_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        image_path = os.path.join(airbus_dir, filename)
        image = cv2.imread(image_path)

        # Resize the image
        resized_image = cv2.resize(image, (image_width, image_height))

        # Save the resized image
        cv2.imwrite(image_path, resized_image)

print("Airbus images resized successfully.")
