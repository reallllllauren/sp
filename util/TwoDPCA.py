import numpy as np

def ttwoDPCA(images,p_1,p_2):
    mean_d = np.mean(images, 0)
    mean_sub = images - mean_d
    no_of_images = images.shape[0]
    mat_height = images.shape[1]
    mat_width = images.shape[2]
    g_t = np.zeros((mat_width, mat_width))
    h_t = np.zeros((mat_height, mat_height))
    for i in range(no_of_images):
        temp = np.dot(mean_sub[i].T, mean_sub[i])
        g_t += temp
        h_t += np.dot(mean_sub[i], mean_sub[i].T)
    g_t /= no_of_images
    h_t /= no_of_images
    d_mat, p_mat = np.linalg.eig(g_t)
    new_bases_gt = p_mat[:, 0:p_1]
    d_mat, p_mat = np.linalg.eig(h_t)
    new_bases_ht = p_mat[:, 0:p_2]
    new_coordinates_temp = np.dot(images, new_bases_gt)
    new_coordinates = np.zeros((no_of_images, p_2, p_1))
    for i in range(no_of_images):
        new_coordinates[i, :, :] = np.dot(new_bases_ht.T, new_coordinates_temp[i])
    return new_coordinates