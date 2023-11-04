import cv2
import numpy as np


image = cv2.imread("C:/Users/hp/Desktop/Computer vision CA/background.jpeg", 0) 

# Define the parameters for speckle noise
speckle_intensity = 0.1 
speckle_image = image + speckle_intensity * image * np.random.normal(0, 1, image.shape)

# Clip the values to ensure they are within the valid range [0, 255]
speckle_image = np.clip(speckle_image, 0, 255).astype(np.uint8)


cv2.imshow('Original Image', image)
cv2.imshow('Speckle Noisy Image', speckle_image)

# Define the threshold value
threshold_value = 200  # Adjust this value as needed

# Apply non-adaptive thresholding to the speckle noisy image
_, thresholded_image = cv2.threshold(speckle_image, threshold_value, 255, cv2.THRESH_BINARY)


cv2.imshow('Thresholded Image', thresholded_image)

cv2.waitKey(0)

cv2.destroyAllWindows()
