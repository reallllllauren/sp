import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def PCA2D_2D(samples, row_top, col_top):
    '''samples are 2d matrices'''
    size = samples[0].shape
    # m*n matrix
    mean = np.zeros(size)

    for s in samples:
        mean = mean + s

    # get the mean of all samples
    mean /= float(len(samples))

    # n*n matrix
    cov_row = np.zeros((size[1],size[1]))
    for s in samples:
        diff = s - mean;
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row /= float(len(samples))
    row_eval, row_evec = np.linalg.eig(cov_row)
    # select the top t evals
    sorted_index = np.argsort(row_eval)
    # using slice operation to reverse
    X = row_evec[:,sorted_index[:-row_top-1 : -1]]

    # m*m matrix
    cov_col = np.zeros((size[0], size[0]))
    for s in samples:
        diff = s - mean;
        cov_col += np.dot(diff,diff.T)
    cov_col /= float(len(samples))
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:,sorted_index[:-col_top-1 : -1]]

    return X, Z


samples = []
for i in range(6):
    im = Image.open('./face_data/'+str(i)+'.jpg')
    im=im.convert("L")
    im_data  = np.empty((im.size[1], im.size[0]))
    for j in range(im.size[1]):
        for k in range(im.size[0]):
            R = im.getpixel((k, j))
            im_data[j,k] = R/255.0
    samples.append(im_data)





def image_transform(top_row,top_col,samples = samples):
    X, Z = PCA2D_2D(samples, top_row, top_col)
    res = np.dot(Z.T, np.dot(samples[0], X))
    res = np.dot(Z, np.dot(res, X.T))

    row_im = Image.new('L', (res.shape[1], res.shape[0]))
    y=res.reshape(1, res.shape[0]*res.shape[1])
    row_im.putdata([int(t*255) for t in y[0].tolist()])
    return row_im

im_10_10 = image_transform(10,10)
im_20_20 = image_transform(20,20)
im_30_30 = image_transform(30,30)
im_40_40 = image_transform(40,40)
im_50_50 = image_transform(50,50)
fig = plt.figure()

a=fig.add_subplot(1,6,1)
imgplot = plt.imshow(im_10_10)
a.set_title('10-10')

a=fig.add_subplot(1,6,2)
imgplot = plt.imshow(im_20_20)
a.set_title('20-20')

a=fig.add_subplot(1,6,3)
imgplot = plt.imshow(im_30_30)
a.set_title('30-30')

a=fig.add_subplot(1,6,4)
imgplot = plt.imshow(im_40_40)
a.set_title('40-40')

a=fig.add_subplot(1,6,5)
imgplot = plt.imshow(im_50_50)
a.set_title('50-50')


im = Image.open('./face_data/0.jpg')
im=im.convert("L")
a=fig.add_subplot(1,6,6)
imgplot = plt.imshow(im)
a.set_title('original\n218-178')

plt.savefig("2d2d_pca.png")