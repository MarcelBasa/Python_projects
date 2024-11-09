import numpy as np
import cv2
import matplotlib.pyplot as plt

def RLE(image):
    last_pixel = image[0]
    encode = []
    count = 0
    for i in range(len(image)):
        if(last_pixel == image[i]):
            count = count + 1
        else:
            encode.append(count)
            encode.append(last_pixel)
            last_pixel = image[i]
            count = 1
    encode.append(count)
    encode.append(last_pixel)
    return encode

def ByteRun(image):
    last_pixel = image[0]
    encode = []
    count = 0
    different_count = 0
    different_pixels = []
    for i in range(len(image)):
        if(last_pixel == image[i]):
            count = count + 1
            if(different_count != 0 ):
                encode.append(different_count*-1)
                encode.extend(different_pixels)
                different_pixels = []
                different_count = 0
        else:
            if(count > 1):
                encode.append(count)
                encode.append(last_pixel) 
                count = 1
            else:
                different_pixels.append(last_pixel)
                different_count = different_count + 1
            last_pixel = image[i]
    if(count > 1):
        encode.append(count)
        encode.append(last_pixel) 
    else:
        different_count = different_count + 1
        different_pixels.append(last_pixel)
        encode.append(different_count*-1)
        encode.extend(different_pixels)
    return encode

def RLE_Decoder(encode):
    decoded = []
    for i in range(0, len(encode), 2):
        count = encode[i]
        pixel = encode[i + 1]
        decoded.extend([pixel] * count)
    return decoded

def ByteRun_Decoder(encode):
    decoded = []
    i = 0
    while i < len(encode):
        if encode[i] < 0: 
            count = abs(encode[i])
            pixels = encode[i + 1:i + count + 1]
            decoded.extend(pixels)
            i += count + 1
        else:
            count = encode[i]
            pixel = encode[i + 1]
            decoded.extend([pixel] * count)
            i += 2
        if i + 1 >= len(encode):
            break 
    return decoded

image = cv2.imread('I:\Systemy multimedialne\lab5\obraz3.jpg')
imageArray = image.flatten()

encodeRLE = RLE(imageArray)
encodeByteRun = ByteRun(imageArray)

decodeRLE = RLE_Decoder(encodeRLE)
decodeByteRun = ByteRun_Decoder(encodeByteRun)

plt.figure()
plt.imshow(image)

plt.figure()
plt.text(0.5, 0.5, 
         "\nCR RLE: {:.4f}".format(len(imageArray) / len(encodeRLE)) +
         "\nCR ByteRun: {:.4f}".format(len(imageArray) / len(encodeByteRun)) +
         "\nPR RLE: {:.2f}%".format((len(encodeRLE) / len(imageArray)) * 100) +
         "\nPR ByteRun: {:.2f}%".format((len(encodeByteRun) / len(imageArray)) * 100),
         horizontalalignment='center', verticalalignment='center')

plt.show()