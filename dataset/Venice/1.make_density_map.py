import mat4py
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import scipy
import scipy.spatial as ss
import re

SOURCE_LABEL_PATH = '/.Venice/venice'
DST_LABEL_PATH = '/.Venice/density_map_init'

def gaussian_filter_density_new(gt,sigma=5):
    sha = (720, 1280)
    density = np.zeros(sha, dtype=np.float32)
    for i in range(len(gt)):
        pt2d = np.zeros(sha, dtype=np.float32)
        try:
            pt2d[gt[i][1],gt[i][0]] = 1.
        except:
            pt2d[gt[i][1]-1,gt[i][0]-1] = 1.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    #print('done.')
    return density

def searchFile(pathname,filename):
    matchedFile =[]
    for root,dirs,files in os.walk(pathname):
        for file in files:
            if re.match(filename,file):
                matchedFile.append((root,file))
    return matchedFile

list1 = searchFile(SOURCE_LABEL_PATH,'(.*).mat')
list1.sort()

for i in range(len(list1)):
    print('{}/{}'.format(i, len(list1)))
    try:
        data = mat4py.loadmat(os.path.join(list1[i][0], list1[i][1]))
    except:
        data = h5py.File(os.path.join(list1[i][0],list1[i][1]), 'r')
    map = gaussian_filter_density_new(data['annotation'],5)
    path = os.path.join(DST_LABEL_PATH,list1[i][1]).replace('.mat','.h5')
    if not os.path.exists(DST_LABEL_PATH):
            os.makedirs(DST_LABEL_PATH)
    with h5py.File(path, 'w') as hf:
            hf['density'] = map
            hf['roi'] = data['roi']
    #break
