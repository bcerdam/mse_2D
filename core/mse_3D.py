from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

def plot_arrays(*args, title='', xlabel='', ylabel='', legends=None, save_path=None):

    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.grid(True, color='white')
    for i, arr in enumerate(args):
        if legends is not None and len(legends) == len(args):
            label = legends[i]
        else:
            label = f'{arr}'
        plt.plot(arr, color=f'C{i}', marker='v', markersize=5, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

# plot_arrays(c_2_0750, p_n_2_0750, w_n_2_0750, xlabel='Scales', ylabel='Entropy', title='MSE 3D (m=2 ; r=0.25*STD)', legends=['Color', 'Pink Noise', 'White Noise'], save_path='/home/bcm/Desktop/Repo/mse_2D/graficos')


def read_images_as_numpy(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            with Image.open(image_path) as img:
                gray_img = img.convert('L')  # convert to grayscale
                np_img = np.asarray(gray_img)
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
    lista_matrices = read_images_as_numpy(X) # Transforma las imagenes a matrices, devuelve una lista con matrices
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

def d_max(lista_matrices, m, i, j, z, a, b, c):
    max_dist = 0
    for n in range(m):
        for k in range(m):
            for l in range(m):
                    dist = abs(lista_matrices[z + n][i + k, j + l] - lista_matrices[c + n][a + k, b + l])
                    if dist > max_dist:
                        max_dist = dist
    return max_dist

def calculate_U_ijz_m(lista_matrices, i, j, z, m, r):
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
    for t in range(1, T-m):
        for a in range(1, H-m):
            for b in range(1, W-m):
                if r == 0:
                    count += 0
                elif d_max(lista_matrices, m, i, j, z, a, b, t) <= r:
                    count += 1
    num_windows = ( (H - m) * (W - m) * (T - m) ) - 1
    return count / num_windows

def calculate_U_ijz_m_plus_one(lista_matrices, i, j, z, m, r):
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
    m += 1
    count = 0
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    for t in range(1, T-m):
        for a in range(1, H-m):
            for b in range(1, W-m):
                if r == 0:
                    count += 0
                elif d_max(lista_matrices, m, i, j, z, a, b, t) <= r:
                    count += 1
    num_windows = ( (H - m) * (W - m) * (T - m) ) - 1
    return count / num_windows

def calculate_U_m(lista_matrices, m, r):
    """
    Calcula el promedio de todos los U_ijz_m
    :param lista_matrices: Lista de matrices, en el caso de video son lista de imagenes (Con las imagenes representadas como matriz)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ijz_m.
    """
    print("Calculating U_m: ")
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    #U_m = np.zeros((H-m, W-m, T-m))
    U_m = np.array([])
    N_m = (H - m) * (W - m) * (T - m)
    for z in range(T - m):
        print(f"U_m time number: {z}")
        for i in range(H - m):
            print(f"U_m row number: {i}")
            for j in range(W - m):
                # U_m[i, j, z] = calculate_U_ijz_m(lista_matrices, i, j, z, m, r)
                U_m = np.append(U_m, calculate_U_ijz_m(lista_matrices, i, j, z, m, r))
    return np.sum(U_m) / N_m

def calculate_U_m_plus_one(lista_matrices, m, r):
    """
    Calcula el promedio de todos los U_ijz_m_plus_one
    :param lista_matrices: Lista de matrices, en el caso de video son lista de imagenes (Con las imagenes representadas como matriz)
    :param m: porte ventana
    :param r: tolerancia de distancia
    :return: devuelve promedio de todos los U_ijz_m_plus_one.
    """
    print("Calculating U_m_plus_one: ")
    H, W = lista_matrices[0].shape
    T = len(lista_matrices)
    #U_m = np.zeros((H-m, W-m, T-m))
    U_m = np.array([])
    N_m = (H - m) * (W - m) * (T - m)
    for z in range(T - m):
        print(f"U_m_plus_one time number: {z}")
        for i in range(H - m):
            print(f"U_m_plus_one row number: {i}")
            for j in range(W - m):
                #U_m[i, j, z] = calculate_U_ijz_m_plus_one(lista_matrices, i, j, z, m, r)
                U_m = np.append(U_m, calculate_U_ijz_m_plus_one(lista_matrices, i, j, z, m, r))
    return np.sum(U_m) / N_m

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
    cube_std = np.std(np.mean(np.array(coarse_graining_Z(path_imagenes, 1)), axis=0))
    for scale in range(1, scales+1):
        coarse_grained = coarse_graining_Z(path_imagenes, scale)
        entropy = calculate_log_ratio(calculate_U_m(coarse_grained, m, r*cube_std), calculate_U_m_plus_one(coarse_grained, m, r*cube_std))
        entropy_values.append(entropy)
    return np.array(entropy_values)



# o = coarse_graining_Z('/home/bcm/Desktop/Repo/mse_2D/datos/3D/white_noise_frames_10x10', 1)
# v = coarse_graining_Z('/home/bcm/Desktop/Repo/mse_2D/datos/3D/white_noise_frames_10x10', 10)
# Image.fromarray(v[0]).show()
# r = 7.5*np.std(np.mean(np.array(o), axis=0))
# z = calculate_U_ijz_m(o, 0, 0, 0, 2, r)
# z_one = calculate_U_ijz_m_plus_one(v, 0, 0, 0, 2, r)
# ratio = z_one / z

#z_m = calculate_U_m(v, 2, 0.25*np.std(np.mean(np.array(v), axis=0)))
# z_m_one = calculate_U_m_plus_one(v, 2, 0.25*np.std(np.mean(np.array(v), axis=0)))


#####

start_time = time.time()
p_n_1_0750 = mse_3D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/3D/pink_noise_frames_10x10', 20, 1, 7.5)
end_time = time.time()
execution_time_p_n_1_0750 = end_time - start_time

#####

start_time = time.time()
w_n_1_0750 = mse_3D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/3D/white_noise_frames_10x10', 20, 1, 7.5)
end_time = time.time()
execution_time_w_n_1_0750 = end_time - start_time

#####

start_time = time.time()
c_1_0750 = mse_3D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/3D/cte_color_frames_10x10', 20, 1, 7.5)
end_time = time.time()
execution_time_c_1_0750 = end_time - start_time

