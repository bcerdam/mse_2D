from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns


def plot_arrays(data_list, title='', xlabel='', ylabel='', legends=None, save_path=None):
    fig, ax = plt.subplots()
    plt.grid('black')
    ax.set_facecolor('lightgrey')
    for i, data_tuple in enumerate(data_list):
        name, data = data_tuple
        if legends is not None and len(legends) == len(data_list):
            label = legends[i]
        else:
            label = name
        plt.plot(data, color=f'C{i}', marker='v', markersize=5, label=label, markeredgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



# plot_arrays(c_2_0750, p_n_2_0750, w_n_2_0750, xlabel='Scales', ylabel='Entropy', title='MSE 3D (m=2 ; r=0.25*STD)', legends=['Color', 'Pink Noise', 'White Noise'], save_path='/home/bcm/Desktop/Repo/mse_2D/graficos')


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

def calculate_mean_matrices(matrices, axis=0):
    return np.mean(matrices, axis=axis)

def plot_image(matrix):
    plt.imshow(matrix, cmap='gray', vmin=0, vmax=255)
    plt.show()

def coarse_graining_Z(X, t):
    '''
    :param X: Path the carpeta con imagenes
    :param t: escala
    :return: lista coarse-graneada
    '''
    lista_matrices = read_images_as_numpy(X, f=False) # Transforma las imagenes a matrices, devuelve una lista con matrices
    coarse_list = []
    if t == 1:
        return lista_matrices
    else:
        for j in range(0, int(len(lista_matrices) / t)):
            lista_imagenes = []
            for i in range(j * t, j * t + t):
                if i >= len(lista_matrices):
                    return coarse_list
                else:
                    i = abs(i)
                    lista_imagenes.append(lista_matrices[i])
            coarse_list.append(calculate_mean_matrices(lista_imagenes)) # Calcula el promedio entre "t" imagenes.
    return coarse_list

def d_max(lista_matrices, m, i, j, k, a, b, c):
    max_dist = 0
    for n in range(m):
        for z in range(m):
            for l in range(m):
                    dist = abs(lista_matrices[k + n][i + z, j + l] - lista_matrices[c + n][a + z, b + l])
                    if dist > max_dist:
                        max_dist = dist
    return max_dist

def calculate_U_ijk_m(lista_matrices, i, j, k, m, r):
    '''
    Calcula la probabilidad de que una ventana (i,j, z) de porte (m*m*m) tenga una diferencia maxima (r) con el resto de las posibles ventanas (a, b, c).

    :param lista_matrices: Lista de matrices, en el caso de video son lista de imagenes (Con las imagenes representadas como matriz)
    :param i: punto inicial de ventana i (fila)
    :param j: punto inicial de ventana j (columna)
    :param j: punto inicial de ventana z (tiempo)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve valor de probabilidad U_ijz_m
    '''
    count = 0
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    N_m = ((H - m) * (W - m) * (T - m))
    for c in range(T - m):
        for a in range(H - m):
            for b in range(W - m):
                if c == k and a == i and b == j:
                    continue
                if d_max(lista_matrices, m, i, j, k, a, b, c) <= r:
                    count += 1
    return count / (N_m-1)

def calculate_U_ijk_m_plus_one(lista_matrices, i, j, k, m, r):
    '''
    Calcula la probabilidad de que una ventana (i,j, z) de porte (m+1*m+1*m+1) tenga una diferencia maxima (r) con el resto de las posibles ventanas (a, b, c).

    :param lista_matrices: Lista de matrices, en el caso de video son lista de imagenes (Con las imagenes representadas como matriz)
    :param i: punto inicial de ventana i (fila)
    :param j: punto inicial de ventana j (columna)
    :param j: punto inicial de ventana z (tiempo)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve valor de probabilidad U_ijz_m
    '''
    count = 0
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    N_m = ((H - m) * (W - m) * (T - m))
    for c in range(T - m):
        for a in range(H - m):
            for b in range(W - m):
                if c == k and a == i and b == j:
                    continue
                if d_max(lista_matrices, m, i, j, k, a, b, c) <= r:
                    count += 1
    return count / (N_m-1)

def calculate_U_m(lista_matrices, m, r):
    """
    Calcula el promedio de todos los U_ijz_m
    :param lista_matrices: Lista de matrices, en el caso de video son lista de imagenes (Con las imagenes representadas como matriz)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ijz_m.
    """
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    U_m = np.array([])
    N_m = (H - m) * (W - m) * (T - m)
    for k in range(T - m):
        print(f"U_m time number: {k}")
        for i in range(H - m):
            for j in range(W - m):
                U_m = np.append(U_m, calculate_U_ijk_m(lista_matrices, i, j, k, m, r))
    return np.sum(U_m) / N_m

def calculate_U_m_plus_one(lista_matrices, m, r):
    """
    Calcula el promedio de todos los U_ijz_m_plus_one
    :param lista_matrices: Lista de matrices, en el caso de video son lista de imagenes (Con las imagenes representadas como matriz)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ijz_m_plus_one.
    """
    m += 1
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    U_m_plus_one = np.array([])
    N_m = (H - m) * (W - m) * (T - m)
    for k in range(T - m):
        print(f"U_m_plus_one time number: {k}")
        for i in range(H - m):
            for j in range(W - m):
                U_m_plus_one = np.append(U_m_plus_one, calculate_U_ijk_m_plus_one(lista_matrices, i, j, k, m, r))
    return np.sum(U_m_plus_one) / N_m

def calculate_log_ratio(Um, Umplus1):
    """
    Calcula entropia para Um y Umplus1
    :param Um: promedio de todos los U_ijz_m
    :param Umplus1: promedio de todos los U_ijz_m_plus_one
    :return: entropia 3D
    """
    if Um == 0:
        return 0
    else:
        U_ratio = Umplus1 / Um
        log_ratio = -np.log(U_ratio)
        return log_ratio

def mse_3D(path_imagenes, scales, m, r):
    entropy_values = []
    valores_pixeles = []
    frames = coarse_graining_Z(path_imagenes, 1)
    for frame in frames:
        for rows in frame:
            for value in rows:
                if value not in valores_pixeles:
                    valores_pixeles.append(value)
    r_parameter = r * np.std(valores_pixeles)

    for scale in range(1, scales+1):
        print(f"Scale: {scale}")
        coarse_grained = coarse_graining_Z(path_imagenes, scale)
        entropy = calculate_log_ratio(calculate_U_m(coarse_grained, m, r_parameter),
                                      calculate_U_m_plus_one(coarse_grained, m, r_parameter))
        entropy_values.append(entropy)
    return np.array(entropy_values)

def parallel_mse_3D(path_imagenes, scale, m, r):
    valores_pixeles = []
    frames = coarse_graining_Z(path_imagenes, 1)
    for frame in frames:
        for rows in frame:
            for value in rows:
                if value not in valores_pixeles:
                    valores_pixeles.append(value)
    r_parameter = r * np.std(valores_pixeles)

    coarse_grained = coarse_graining_Z(path_imagenes, scale)
    entropy = calculate_log_ratio(calculate_U_m(coarse_grained, m, r_parameter),
                                  calculate_U_m_plus_one(coarse_grained, m, r_parameter))
    print(f"Scale: {scale}")
    return entropy

# Para probar algoritmo

# # White Noise Frames
#
# start_time = time.time()
# w_n_1_050 = mse_3D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/3D/white_noise_frames_50x50', 20, 1, 0.5)
# end_time = time.time()
# execution_time_w_n_1_050 = end_time - start_time



