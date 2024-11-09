import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def u_law_kompresja(sygnał, skala=255):
    u = skala
    return np.sign(sygnał) * (np.log(1 + u * np.abs(sygnał)) / np.log(1 + u))

def u_law_kompresja_kwant(sygnał, skala=255, liczba_poziomów=256):
    u = skala
    sygnał_kwantyzowany = np.round(sygnał * liczba_poziomów) / liczba_poziomów
    return np.sign(sygnał_kwantyzowany) * (np.log(1 + u * np.abs(sygnał_kwantyzowany)) / np.log(1 + u))

def u_law_dekompresja(sygnał, skala=255):
    return np.sign(sygnał) * (1 / skala) * ((1 + skala) ** np.abs(sygnał) - 1)

def DPCM_kompresja(x, skala=255):
    y = np.zeros(x.shape)
    e = 0
    for i in range(x.shape[0]):
        y[i] = x[i] - e
        e += y[i]
    return y

def DPCM_kompresja_kwant(x, skala=255, liczba_poziomów=256):
    y = np.zeros(x.shape)
    e = 0
    for i in range(x.shape[0]):
        x_kwantyzowany = np.round(x[i] * liczba_poziomów) / liczba_poziomów
        y[i] = x_kwantyzowany - e
        e += y[i]
    return y


def DPCM_decompress(y, skala=255):
    x = np.zeros(y.shape)
    e = 0
    for i in range(y.shape[0]):
        x[i] = y[i] + e
        e += x[i]
    return x

x = np.linspace(-1, 1, 1000)
sygnał = 0.9 * np.sin(np.pi * x * 4)

sygnał_kompresowany_u_law = u_law_kompresja(sygnał)
sygnał_kompresowany_u_law_kwant = u_law_kompresja_kwant(sygnał, 255, 6)
sygnał_dekompresowany_u_law = u_law_dekompresja(sygnał_kompresowany_u_law)

sygnał_kompresowany_DPCM = DPCM_kompresja(sygnał)
sygnał_kompresowany_DPCM_kwant = DPCM_kompresja_kwant(sygnał)
sygnał_dekompresowany_DPCM = DPCM_decompress(sygnał_kompresowany_DPCM)

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.plot(x, sygnał)
plt.title('Sygnał oryginalny')
plt.xlabel('Czas')
plt.ylabel('Amplituda')

plt.subplot(2, 3, 2)
plt.plot(x, sygnał_kompresowany_u_law)
plt.title('Sygnał skompresowany (u-law)')
plt.xlabel('Czas')
plt.ylabel('Amplituda')

plt.subplot(2, 3, 3)
plt.plot(x, sygnał_kompresowany_u_law_kwant)
plt.title('Sygnał skompresowany (u-law) 6-bit')
plt.xlabel('Czas')
plt.ylabel('Amplituda')

plt.subplot(2, 3, 4)
plt.plot(x, sygnał_kompresowany_DPCM)
plt.title('Sygnał skompresowany (DPCM)')
plt.xlabel('Czas')
plt.ylabel('Amplituda')

plt.subplot(2, 3, 5)
plt.plot(x, sygnał_kompresowany_DPCM_kwant)
plt.title('Sygnał skompresowany (DPCM) 6bit')
plt.xlabel('Czas')
plt.ylabel('Amplituda')
plt.tight_layout()
plt.show()