import os
import os.path as osp
import scipy.io as scio
from PIL import Image
import pickle

import numpy as np
import SimpleITK as sitk

class Obj:
    def __init__(self, filename, id) -> None:
        self.filename = filename
        self.id = id

def readAnnotation(root_dir):
    objset = []
    
    count = 0
    img_ids = []
    id_name_pair = {}
    with open(osp.join(root_dir, 'images.txt'), 'r') as f:
        raw = [tmp.replace('\n', '') for tmp in f.readlines()]
        for tmp in raw:
            id, img_name = tmp.split(' ')
            id_name_pair[id] = img_name
            count += 1

    for i in range(1, count+1, 1):
        img_ids.append(str(i))

    for idx, id in enumerate(img_ids):
        filename = id_name_pair[id]
        objset.append(Obj(filename, idx))
    
    return objset

def getImg(obj : Obj, img_dir, img_size):
    img_full_path = osp.join(img_dir, obj.filename)
    img = np.array(Image.open(img_full_path))
    img_crop = Image.fromarray(img).resize(img_size, resample=Image.BICUBIC)
    img_crop = np.array(img_crop)
    if img_crop.ndim == 2:
        img_crop = img_crop[..., np.newaxis]
    return img_crop

def norm_img(img):
    max_v = img.max()
    min_v = img.min()
    img = (img - min_v) / (max_v - min_v)
    return img*255

def main():
    root_dir = r'D:\datasets\Sliver07\Sliver2d\forEP'
    ori_img_dir = r'D:\datasets\Sliver07\Sliver2d\forEP\ct'
    objset = readAnnotation(root_dir)
    objnum = len(objset)
    img_size = (256, 256)
    ori_imgs = []
    for i in range(objnum):
        img = getImg(objset[i], ori_img_dir, img_size)
        ori_imgs.append(img)
    
    # [n, h, w, d]
    _imgs = np.stack(ori_imgs, axis=0)
    with open('Sliver2d/images.pkl', 'wb+') as f:
        pickle.dump(
        {
            'images': _imgs
        },
        f
    )

    # [h, w, d, n]
    _imgs = np.stack(ori_imgs, axis=3)
    scio.savemat(
        'Sliver2d/images.mat',
        {
            'images': _imgs
        }
    )
    # sitk.WriteImage(sitk.GetImageFromArray(_imgs[..., 0, :].transpose(2, 0, 1)), 'BTCV/images.nii')

if __name__ == '__main__':
    main()