import numpy as np
import cv2
import scipy.fftpack
import matplotlib.pyplot as plt

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

class ver1:
    def __init__(self):
        self.Y = np.array([])
        self.Cb = np.array([])
        self.Cr = np.array([])
        self.ChromaRatio = "4:4:4"
        self.QY = np.ones((8, 8))
        self.QC = np.ones((8, 8))
        self.shape = (0, 0, 3)

def CompressJPEG(RGB, Ratio="4:4:4", QY=np.ones((8,8)), QC=np.ones((8,8))):
    JPEG = ver1()

    YCrCb = cv2.cvtColor(RGB.astype(np.uint8), cv2.COLOR_RGB2YCrCb).astype(int)
    YCrCb[:,:,0] -= 128  

    JPEG.shape = RGB.shape
    JPEG.Y = YCrCb[:,:,0]
    JPEG.Cb = YCrCb[:,:,1]
    JPEG.Cr = YCrCb[:,:,2]

    JPEG.Y = CompressLayer(JPEG.Y, QY)
    JPEG.Cr = CompressLayer(JPEG.Cr, QC)
    JPEG.Cb = CompressLayer(JPEG.Cb, QC)

    return JPEG

def DecompressJPEG(JPEG):
    Y, Cr, Cb = DecompressLayer(JPEG.Y, JPEG.QY, JPEG.shape)

    YCrCb = np.zeros(JPEG.shape, dtype=int)
    YCrCb[:,:,0] = Y
    YCrCb[:,:,1] = Cr
    YCrCb[:,:,2] = Cb
    YCrCb[:,:,0] += 128  

    original_data_rgb = cv2.cvtColor(YCrCb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    
    return original_data_rgb


def CompressBlock(block, Q):
    return block.flatten()

def DecompressBlock(vector, Q):
    if vector.size != 64:
        print(f"Invalid vector size: {vector.size}.")
        return np.zeros((8, 8), dtype=int) 
    
    return vector.reshape((8, 8)) 


def CompressLayer(L, Q):
    S = np.array([])
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w:(w+8), k:(k+8)]
            S = np.append(S, CompressBlock(block, Q))
    return S

def DecompressLayer(S, Q, original_shape):
    L = np.zeros(original_shape, dtype=int)  

    m = original_shape[1] // 8  
    n = original_shape[0] // 8  

    for idx, i in enumerate(range(0, S.shape[0], 64)):
        vector = S[i:(i+64)]  
        block = DecompressBlock(vector, Q)  

        k = (idx % m) * 8
        w = (idx // m) * 8

        if w + 8 <= original_shape[0] and k + 8 <= original_shape[1]:
            for i in range(8):
                for j in range(8):
                    L[w + i, k + j] = block[i, j]
    
    Y = L[:,:,0]
    Cr = L[:,:,1]
    Cb = L[:,:,2]

    return Y, Cr, Cb

def zigzag(A):
    template = np.array([
            [0,  1,  5,  6,  14, 15, 27, 28],
            [2,  4,  7,  13, 16, 26, 29, 42],
            [3,  8,  12, 17, 25, 30, 41, 43],
            [9,  11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
            ])
    if len(A.shape) == 1:
        B = np.zeros((8, 8))
        for r in range(0, 8):
            for c in range(0, 8):
                B[r, c] = A[template[r, c]]
    else:
        B = np.zeros((64,))
        for r in range(0, 8):
            for c in range(0, 8):
                B[template[r, c]] = A[r, c]
    return B

def ChromaSubsampling(JPEG, ratio="4:4:4"):
    if ratio == "4:2:2":
        JPEG.Cb = JPEG.Cb[:, ::2]
        JPEG.Cr = JPEG.Cr[:, ::2]
        JPEG.ChromaRatio = ratio
    elif ratio == "4:2:0":
        # Redukcja chrominancji 4:2:0
        ## Nie dla laboratoriów z JPEG
        pass
    else:  # Domyślnie: "4:4:4"
        # Brak redukcji chrominancji (bez zmian)
        pass
    return JPEG

def display_images_before_and_after_compression(original_image_rgb, reconstructed_image_rgb):
    fig, axs = plt.subplots(4, 2, sharey=True)
    fig.set_size_inches(9, 13)

    # Obraz oryginalny
    axs[0, 0].imshow(original_image_rgb)  # RGB
    original_ycrcb = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2YCrCb)
    axs[1, 0].imshow(original_ycrcb[:, :, 0], cmap=plt.cm.gray)
    axs[2, 0].imshow(original_ycrcb[:, :, 1], cmap=plt.cm.gray)
    axs[3, 0].imshow(original_ycrcb[:, :, 2], cmap=plt.cm.gray)

    # Obraz po dekompresji
    axs[0, 1].imshow(reconstructed_image_rgb)  # RGB
    reconstructed_ycrcb = cv2.cvtColor(reconstructed_image_rgb, cv2.COLOR_RGB2YCrCb)
    axs[1, 1].imshow(reconstructed_ycrcb[:, :, 0], cmap=plt.cm.gray)
    axs[2, 1].imshow(reconstructed_ycrcb[:, :, 1], cmap=plt.cm.gray)
    axs[3, 1].imshow(reconstructed_ycrcb[:, :, 2], cmap=plt.cm.gray)

    plt.show()

def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

def calculate_vector_size(image_shape):
    return image_shape[0] * image_shape[1] // 64

original_image = cv2.imread("BIG_0001.jpg")
PRZED_RGB = original_image
PO_RGB = original_image

display_images_before_and_after_compression(PRZED_RGB, PO_RGB)

JPEG = CompressJPEG(original_image)

# Obliczenie rozmiaru oryginalnych danych
original_vector_size = calculate_vector_size(original_image.shape)

# Obliczenie rozmiaru skompresowanych danych
compressed_Y_size = JPEG.Y.shape[0]
compressed_Cb_size = JPEG.Cb.shape[0]
compressed_Cr_size = JPEG.Cr.shape[0]

# Obliczenie stopnia kompresji dla każdej warstwy
compression_ratio_Y = calculate_compression_ratio(original_vector_size, compressed_Y_size)
compression_ratio_Cb = calculate_compression_ratio(original_vector_size, compressed_Cb_size)
compression_ratio_Cr = calculate_compression_ratio(original_vector_size, compressed_Cr_size)

print("Stopień kompresji dla warstwy Y:", compression_ratio_Y)
print("Stopień kompresji dla warstwy Cb:", compression_ratio_Cb)
print("Stopień kompresji dla warstwy Cr:", compression_ratio_Cr)
