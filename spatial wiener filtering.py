import cv2
import numpy as np

# Load the original image
image = cv2.imread("C:/Users/hp/Desktop/Computer vision CA/coin_testing.png")

# Define parameters for stronger Gaussian noise
mean = 0
stddev = 80

# Generate Gaussian noise with the specified mean and standard deviation
noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)

# Add the Gaussian noise to the original image
noisy_image = cv2.add(image, noise)

# Display the original and noisy images
cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)

# Apply Wiener filter for denoising
denoised_image = cv2.fastNlMeansDenoisingColored(noisy_image, None, 20, 20, 5, 11)

# Display the denoised image
cv2.imshow('Denoised Image', denoised_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
