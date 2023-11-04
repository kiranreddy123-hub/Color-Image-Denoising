import cv2
import numpy as np


image = cv2.imread("C:/Users/hp/Desktop/Computer vision CA/background.jpeg", 0)

# Define the parameters for speckle noise
speckle_intensity = 0.15  # Adjust this value to control the noise intensity
speckle_image = image + speckle_intensity * image * np.random.normal(0, 1, image.shape)

# Clip the values to ensure they are within the valid range [0, 255]
speckle_image = np.clip(speckle_image, 0, 255).astype(np.uint8)

# Display the original image and the speckle noisy image
cv2.imshow('Original Image', image)
cv2.imshow('Speckle Noisy Image', speckle_image)

# Define the parameters for adaptive thresholding
block_size = 11  # Size of the neighborhood for thresholding
c = 2  # Constant subtracted from the mean

# Apply adaptive thresholding to the speckle noisy image
adaptive_thresh = cv2.adaptiveThreshold(speckle_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)

# Display the adaptive thresholded image
cv2.imshow('Adaptive Thresholded Image', adaptive_thresh)

cv2.waitKey(0)

cv2.destroyAllWindows()
