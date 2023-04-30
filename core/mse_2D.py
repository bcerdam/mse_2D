import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
import time
import os

def read_images_as_numpy(folder_path, f):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                gray_img = img.convert('L')  # convert to grayscale
                np_img = np.asarray(gray_img)
                if f == True:
                    images.append((np_img, filename))
                else:
                    images.append(np_img)
    return images

def image_to_array(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    return arr

def plot_image_as_array(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    return image

def coarse_graining_2D(image, scale):
    rows, cols = image.shape
    new_rows = rows // scale
    new_cols = cols // scale
    coarse_grained = np.zeros((new_rows, new_cols))
    for i in range(new_rows):
        for j in range(new_cols):
            coarse_grained[i, j] = np.mean(image[i*scale:(i+1)*scale, j*scale:(j+1)*scale])
    return coarse_grained

def d_max(image, m, i, j, a, b):
    max_dist = 0
    for k in range(m):
        for l in range(m):
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
    N_m = (H - m) * (W - m)
    for a in range(H - m):
        for b in range(W - m):
            if a == i and b == j:
                continue
            if d_max(image, m, i, j, a, b) <= r:
                count += 1
    return count / (N_m-1)

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
    count = 0
    H, W = image.shape
    N_m = (H - m) * (W - m)
    for a in range(H - m):
        for b in range(W - m):
            if a == i and b == j:
                continue
            if d_max(image, m, i, j, a, b) <= r:
                count += 1
    return count / (N_m-1)

def calculate_U_m(image, m, r):
    """
    Calcula el promedio de todos los U_ij_m
    :param image: Imagen en formato array coarse-graneada
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ij_m.
    """
    H, W = image.shape
    U_m = np.zeros((H-m, W-m))
    N_m = (H - m) * (W - m)
    for i in range(H - m):
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
    m += 1
    H, W = image.shape
    U_m_plus_one = np.zeros((H-m, W-m))
    N_m = (H - m) * (W - m)
    for i in range(H - m):
        for j in range(W - m):
            U_m_plus_one[i, j] = calculate_U_ij_m_plus_one(image, i, j, m, r)
    return np.sum(U_m_plus_one) / N_m

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
    pixel_values = []
    for row in image_array:
        for value in row:
            if value not in pixel_values:
                pixel_values.append(value)

    r_parameter = r * np.std(pixel_values)
    for scale in range(1, scales+1):
        print(f"Scale: {scale}")
        coarse_grained = coarse_graining_2D(image_array, scale)
        entropy = calculate_log_ratio(calculate_U_m(coarse_grained, m, r_parameter),
                                      calculate_U_m_plus_one(coarse_grained, m, r_parameter))
        entropy_values.append(entropy)
    return np.array(entropy_values)

def parallel_mse_2D(image, scale, m, r):
    print(scale)
    pixel_values = []
    for row in image:
        for value in row:
            if value not in pixel_values:
                pixel_values.append(value)
    r_parameter = r * np.std(pixel_values)
    coarse_grained = coarse_graining_2D(image, scale)
    entropy = calculate_log_ratio(calculate_U_m(coarse_grained, m, r_parameter),
                                  calculate_U_m_plus_one(coarse_grained, m, r_parameter))
    return entropy

# Para probar algoritmo en general:

# print('Bee Hive Image')
#
# start_time = time.time()
# b_2_025 = mse_2D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/2D/100x100_test/bee_hive.jpg', 20, 2, 0.25)
# end_time = time.time()
# execution_time_b_2_025 = end_time - start_time




