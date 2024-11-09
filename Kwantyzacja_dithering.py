import numpy as np
import cv2
import matplotlib.pyplot as plt

def colorFit(pixel_color, colors):
        color_difference = np.linalg.norm((pixel_color/255) - colors, axis=1)
        closest_color_index = np.argmin(color_difference)
        return colors[closest_color_index]

def dithering_losowy(image):
        rows = image.shape[0] 
        cols = image.shape[1]
        random_matrix = np.random.randint(0, 256, size=(rows, cols, 1))
        binary_image = (image >= random_matrix).astype(int)
        return binary_image

def dithering_zorganizowany(image, threshold_map, color_palette):
        rows = image.shape[0]
        cols = image.shape[1]
        size = threshold_map.shape[0]
        quantized_image = np.zeros((rows, cols, 3), dtype=np.uint8)
        for i in range(rows):
                for j in range(cols):
                        threshold_index = (i % size, j % size)
                        temp_value = image[i, j] + threshold_map[threshold_index]
                        quantized_pixel = colorFit(temp_value, color_palette)*255
                        quantized_image[i, j] = quantized_pixel
        return quantized_image

def dithering_floyd_steinberg(image, color_palette):
        rows = image.shape[0]
        cols = image.shape[1]
        dithered_image = np.zeros_like(image)
        for y in range(rows):
                for x in range(cols):
                        old_pixel = image[y, x]
                        new_pixel = colorFit(old_pixel, color_palette)*255
                        dithered_image[y, x] = new_pixel 
                        quant_error = old_pixel - new_pixel
                        if x + 1 < cols:
                                error = quant_error * 7 / 16
                                dithered_image[y, x + 1] += error.astype(np.uint8)
                        if x - 1 >= 0 and y + 1 < rows:
                                error = quant_error * 3 / 16
                                dithered_image[y + 1, x - 1] += error.astype(np.uint8)
                        if y + 1 < rows:
                                error = quant_error * 5 / 16
                                dithered_image[y + 1, x] += error.astype(np.uint8)
                        if x + 1 < cols and y + 1 < rows:
                                error = quant_error * 1 / 16
                                dithered_image[y + 1, x + 1] += error.astype(np.uint8)
        return dithered_image

# Wczytanie obrazu
image = cv2.imread('I:\Systemy multimedialne\lab4\IMG_SMALL\SMALL_0006.jpg')
img_kwant = np.zeros_like(image)
dithering_losowy_image = np.zeros_like(image)
dithering_zorganizowany_image = np.zeros_like(image)
dithered_floyd_steinberg_image = np.zeros_like(image)
pallet8 = np.array([
        [0.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0,],
        [0.0, 1.0, 0.0,],
        [0.0, 1.0, 1.0,],
        [1.0, 0.0, 0.0,],
        [1.0, 0.0, 1.0,],
        [1.0, 1.0, 0.0,],
        [1.0, 1.0, 1.0,],
])
pallet16 =  np.array([
        [0.0, 0.0, 0.0,], 
        [0.0, 1.0, 1.0,],
        [0.0, 0.0, 1.0,],
        [1.0, 0.0, 1.0,],
        [0.0, 0.5, 0.0,], 
        [0.5, 0.5, 0.5,],
        [0.0, 1.0, 0.0,],
        [0.5, 0.0, 0.0,],
        [0.0, 0.0, 0.5,],
        [0.5, 0.5, 0.0,],
        [0.5, 0.0, 0.5,],
        [1.0, 0.0, 0.0,],
        [0.75, 0.75, 0.75,],
        [0.0, 0.5, 0.5,],
        [1.0, 1.0, 1.0,], 
        [1.0, 1.0, 0.0,]
])

# ColorFit
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        img_kwant[i,j]=colorFit(image[i,j], pallet8)*255

# Dithering losowy
dithering_losowy_image = dithering_losowy(image)*255

# Dithering Zorganizowany
threshold_map  = np.array([[-0.25, 0.25],
                          [0.5, 0]])*255
dithering_zorganizowany_image = dithering_zorganizowany(image, threshold_map, pallet8)

# Dithering Floyd Steinberg
dithered_floyd_steinberg_image = dithering_floyd_steinberg(image, pallet8)

plt.subplot(2,3,1)
plt.imshow(image)
plt.title("orginalny")
plt.subplot(2,3,2)
plt.imshow(img_kwant)
plt.title("kwantyzacjca")
plt.subplot(2,3,3)
plt.imshow(dithering_losowy_image)
plt.title("Dithering losowy")
plt.subplot(2,3,4)
plt.imshow(dithering_zorganizowany_image)
plt.title("Dithering zorganizowany")
plt.subplot(2,3,5)
plt.imshow(dithered_floyd_steinberg_image)
plt.title("Dithering Floyd Steinberg")

plt.suptitle("Dithering paleta 8 kolorów")

plt.axis("on")
plt.show()
