import numpy as np
import seaborn as sns
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def image_to_array(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    return arr

def plot_image_as_array(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    return image

def coarse_grain_image_mean(image, scale):
    rows, cols = image.shape
    new_rows = rows // scale
    new_cols = cols // scale
    coarse_grained = np.zeros((new_rows, new_cols))
    for i in range(new_rows):
        for j in range(new_cols):
            coarse_grained[i, j] = np.mean(image[i*scale:(i+1)*scale, j*scale:(j+1)*scale])
    return coarse_grained

# def d_max(image, m, i, j, a, b):
#     u = image[i:i+m, j:j+m]
#     v = image[a:a+m, b:b+m]
#     return np.max(np.abs(u - v))

def d_max(image, m, i, j, a, b):
    max_dist = 0
    for k in range(m-1):
        for l in range(m-1):
            dist = abs(image[i+k, j+l] - image[a+k, b+l])
            if dist > max_dist:
                max_dist = dist
    return max_dist

def calculate_U_ij_m(image, i, j, m, r):
    '''
    Calcula la probabilidad de que una ventana (i,j) de porte (m*m) tenga una diferencia maxima (r) con el resto de las posibles
    ventanas (a, b).

    :param image: Imagen en formato array coarse-graneada
    :param i: punto inicial de ventana i (fila)
    :param j: punto inicial de ventana j (columna)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve valor de probabilidad U_ij_m
    '''
    count = 0
    H, W = image.shape
    for a in range(1, H-m):
        for b in range(1, W-m):
            if d_max(image, m, i, j, a, b) < r:
                count += 1
    return count / (np.prod(image[:-m, :-m].shape) - 1)

def calculate_U_ij_m_plus_one(image, i, j, m, r):
    '''
    Calcula la probabilidad de que una ventana (i,j) de porte (m+1*m+1) tenga una diferencia maxima (r) con el resto de las posibles
    ventanas (a, b).

    :param image: Imagen en formato array coarse-graneada
    :param i: punto inicial de ventana i (fila)
    :param j: punto inicial de ventana j (columna)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve valor de probabilidad U_ij_m_plus_one
    '''
    m += 1
    count = 0
    H, W = image.shape
    for a in range(1, H-m):
        for b in range(1, W-m):
            if d_max(image, m, i, j, a, b) < r:
                count += 1
    return count / (np.prod(image[:-m, :-m].shape) - 1)

def calculate_U_m(image, m, r):
    """
    Calcula el promedio de todos los U_ij_m
    :param image: Imagen en formato array coarse-graneada
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ij_m.
    """
    print("Calculating U_m: ")
    H, W = image.shape
    U_m = np.zeros((H-m, W-m))
    N_m = (H - m) * (W - m)
    for i in range(H - m):
        print(f"U_m row number: {i}")
        for j in range(W - m):
            U_m[i, j] = calculate_U_ij_m(image, i, j, m, r)
    return np.sum(U_m) / N_m

def calculate_U_m_plus_one(image, m, r):
    """
    Calcula el promedio de todos los U_ij_m_plus_one
    :param image: Imagen en formato array coarse-graneada
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ij_m_plus_one.
    """
    print("Calculating U_m_plus_one: ")
    H, W = image.shape
    U_m = np.zeros((H-m, W-m))
    N_m = (H - m) * (W - m)
    for i in range(H - m):
        print(f"U_m_plus_one row number: {i}")
        for j in range(W - m):
            U_m[i, j] = calculate_U_ij_m_plus_one(image, i, j, m, r)
    return np.sum(U_m) / N_m

def calculate_log_ratio(Um, Umplus1):
    """
    Calcula SampEn2D
    :param Um: promedio de todos los U_ij_m
    :param Umplus1: promedio de todos los U_ij_m_plus_one
    :return: Sample Entropy 2D
    """
    if Um == 0:
        return 0
    else:
        U_ratio = Umplus1 / Um
        log_ratio = -np.log(U_ratio)
        return log_ratio


def mse_2D(image, scales, m, r):
    image_array = image_to_array(image)
    entropy_values = []
    for scale in range(1, scales+1):
        coarse_grained = coarse_grain_image_mean(image_array, scale)
        entropy = calculate_log_ratio(calculate_U_m(coarse_grained, m, r*np.std(image_array)),
                                      calculate_U_m_plus_one(coarse_grained, m, r*np.std(image_array)))
        entropy_values.append(entropy)
    return np.array(entropy_values)

# Para probar funciones especificas:

# v = calculate_U_ij_m(image_to_array('white_noise_3.png'), 0, 0, 2, 3*np.std(image_to_array('white_noise_3.png')))
# v = calculate_U_m(image_to_array('white_noise_3.png'), 2, 3*np.std(image_to_array('white_noise_3.png')))

# Para probar algoritmo en general:

# white_noise_image = mse_2D('/home/bcm/Desktop/Repo/mse_2D/datos/100x100_test/white_noise_3.png', 20, 2, 0.25)
# color_image = mse_2D('/home/bcm/Desktop/Repo/mse_2D/datos/100x100_test/solid-color-image.png', 20, 2, 0.25)
# nature_fractal_image = mse_2D('/home/bcm/Desktop/Repo/mse_2D/datos/100x100_test/nature_fractal.png', 20, 2, 0.25)

