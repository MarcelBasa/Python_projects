import numpy as np
import cv2
import matplotlib.pyplot as plt

def nearest_neighbor_interpolation(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            x = min(int(i / scale_factor), height - 1)
            y = min(int(j / scale_factor), width - 1)
            new_image[i, j] = image[x, y]
    
    return new_image

def bilinear_interpolation(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_factor
            y = j / scale_factor
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
            dx, dy = x - x1, y - y1
            new_image[i, j] = (1 - dx) * (1 - dy) * image[x1, y1] + dx * (1 - dy) * image[x2, y1] + (1 - dx) * dy * image[x1, y2] + dx * dy * image[x2, y2]
    
    return new_image

def reduce_image_average(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            x_start, x_end = int(i / scale_factor), min(int((i + 1) / scale_factor), height)
            y_start, y_end = int(j / scale_factor), min(int((j + 1) / scale_factor), width)
            new_image[i, j] = np.mean(image[x_start:x_end, y_start:y_end], axis=(0, 1))
    
    return new_image

def reduce_image_weighted_average(image, scale_factor, weights=None):
    if weights is None:
        weights = [1] * (int(scale_factor)**2)
    
    weights_sum = sum(weights)
    normalized_weights = [weight / weights_sum for weight in weights]
    
    height, width, channels = image.shape
    new_height = int(height / scale_factor)
    new_width = int(width / scale_factor)
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    if len(normalized_weights) != channels:
        raise ValueError("Length of weights must be equal to the number of image channels.")
    
    for i in range(0, height, int(scale_factor)):
        for j in range(0, width, int(scale_factor)):
            x_start, x_end = i, min(i + int(scale_factor), height)
            y_start, y_end = j, min(j + int(scale_factor), width)
            for c in range(channels):
                new_image[i//int(scale_factor), j//int(scale_factor), c] = np.average(image[x_start:x_end, y_start:y_end, c], weights=normalized_weights).astype(np.uint8)
    
    return new_image

def reduce_image_median(image, scale_factor):
    height, width, channels = image.shape
    new_height = int(height / scale_factor)
    new_width = int(width / scale_factor)
    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    for i in range(0, height, scale_factor):
        for j in range(0, width, scale_factor):
            x_start, x_end = i, min(i + scale_factor, height)
            y_start, y_end = j, min(j + scale_factor, width)
            new_image[i//scale_factor, j//scale_factor] = np.median(image[x_start:x_end, y_start:y_end], axis=(0, 1))
    
    return new_image

def resize_image(image, scale_factor):
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Wczytanie obrazów
image_orginal = cv2.imread('I:\Systemy multimedialne\lab3\IMG_BIG\BIG_0002.jpg')

# Wczytanie obrazu
image = cv2.imread('I:\Systemy multimedialne\lab3\IMG_BIG\BIG_0002.jpg')

# Przykładowe wywołania funkcji
image_method = reduce_image_average(image, 0.5)
resized_image_orginal = resize_image(image, 0.05)  
resized_image_method = resize_image(image_method, 0.05)  

plt.subplot(2,2,1)
plt.imshow(image_orginal)
plt.title("img")
plt.subplot(2,2,2)
plt.imshow(image_method)
plt.title("metoda: średniej ważonej")
plt.subplot(2,2,3)
plt.imshow(resized_image_orginal)
plt.title("resize")
plt.subplot(2,2,4)
plt.imshow(resized_image_method)
plt.title("resize")

plt.axis("on")
plt.show()

#nearest_neighbor_interpolation(image, 0.5)
#bilinear_interpolation(image, 0.5)
#reduce_image_average(image, 0.5)
#reduce_image_weighted_average(image, 0.5)
#reduce_image_median(image, 2)