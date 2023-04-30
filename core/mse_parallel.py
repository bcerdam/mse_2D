import mse_2D
import mse_3D
import multiprocessing as mp
import os
import time

def parallel_mse_2D(folder_path, scales, m, r):
    imagenes = mse_2D.read_images_as_numpy(folder_path, f=True)
    inputs = []
    names = []
    for imagen in imagenes:
        names.append(imagen[1])
        for scale in range(1, scales+1):
            inputs.append([imagen[0], scale, m, r])

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(mse_2D.parallel_mse_2D, [*inputs])
    final_result = list(zip(names, [results[i:i+scales] for i in range(0, len(results), scales)]))
    return final_result



def parallel_mse_3D(folder_path, scales, m, r):
    inputs = []
    names = []
    for filename in os.listdir(folder_path):
        if filename == '.DS_Store':
            continue
        else:
            names.append(filename)
            for scale in range(1, scales+1):
                inputs.append([folder_path+'/'+filename, scale, m, r])

    start_time = time.time()

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(mse_3D.parallel_mse_3D, [*inputs])


    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)

    final_result = list(zip(names, [results[i:i + scales] for i in range(0, len(results), scales)]))
    return final_result

# EJEMPLO DE COMO CORRER
# IMPORTANTE: Hay que correr el codigo de abajo por la consola de python, si no tira un error (No se porque).

#2D

# v = parallel_mse_2D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/2D/100x100_bee_hives', 20, 2, 0.5)

# Esto retorna una lista de listas de esta estructura:
# [(nombre_imagen_1, [valores_entropia_nombre_imagen_1]),..., (nombre_imagen_n, [valores_entropia_nombre_imagen_n])],

#3D

# v = parallel_mse_3D('/Users/brunocerdamardini/Desktop/repo/mse_2D/datos/3D/10x10', 20, 1, 0.5)

# Esto retorna una lista de listas de esta estructura:
# [(nombre_frames_1, [valores_entropia_nombre_frames_1]),..., (nombre_frames_n, [valores_entropia_nombre_frames_n])],