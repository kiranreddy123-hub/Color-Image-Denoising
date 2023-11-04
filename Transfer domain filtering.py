import cv2
import numpy as np


image_path = "C:/Users/hp/Desktop/Computer vision CA/face_testing.jpeg"


image = cv2.imread(image_path, 0)  

if image is None:
    print(f"Failed to load the image at '{image_path}'. Please check the file path and format.")
else:
    # Define the probability of adding salt and pepper noise
    noise_prob = 0.05  

    # Create a mask of random values with the same data type as the image
    noise = np.random.choice([0, 255], size=image.shape, p=[1 - noise_prob/2, noise_prob/2]).astype(np.uint8)

  
    noisy_image = cv2.add(image, noise)

    cv2.imwrite('noisy_image.jpg', noisy_image)

    # Apply Fourier domain filtering for denoising
    f_transform = np.fft.fft2(noisy_image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Define a circular mask for the low-pass filter
    rows, cols = noisy_image.shape
    crow, ccol = rows // 2, cols // 2
    filter_radius = 30  
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), filter_radius, 1, -1)

    # Apply the mask to the Fourier transformed image
    f_transform_filtered = f_transform_shifted * mask

    # Inverse Fourier transform to get the denoised image
    #denoised_image = np.abs(np.fft.ifft2(np.fft.ifftshift(f_transform_filtered)))
    denoised_image = cv2.medianBlur(noisy_image, 3)

    cv2.imwrite('denoised_image.jpg', denoised_image)