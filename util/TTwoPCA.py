import numpy as np

def twoDPCA(images,p):
    mean_d = np.mean(images, 0)
    mean_sub = images - mean_d
    no_of_images = images.shape[0]
    mat_height = images.shape[2]
    g_t = np.zeros((mat_height, mat_height))
    for i in range(no_of_images):
        temp = np.dot(mean_sub[i].T, mean_sub[i])
        g_t += temp
    g_t /= no_of_images
    d_mat, p_mat = np.linalg.eig(g_t)
    new_bases = p_mat[:, 0:p]
    print(new_bases.shape)
    new_coordinates = np.dot(images, new_bases)
    return new_coordinates