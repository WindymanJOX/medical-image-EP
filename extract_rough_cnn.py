import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
import scipy.io as scio

from extract_net_parameters import getNetParameters
from models import *

def getStat(data):
    cd = data.shape[3]
    mean = np.zeros(cd)
    std = np.zeros(cd)
    for i in range(data.shape[0]):
        for d in range(cd):
            mean[d] += data[i, :, :, d].mean()
            std[d] += data[i, :, :, d].std()
    mean /= data.shape[0]
    std /= data.shape[0]
    return (mean, std)

def getObjFeatures(image, net, stat):
    image = np.array(image, dtype=np.float32)
    trans = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.to(torch.float32) / 255.0
    ])
    # [1, 1, 256, 256]
    image = trans(image).cuda().unsqueeze(0)
    with torch.no_grad():
        _ = net(image)
    return net.getFeatures()

def getRoughCNN(net, conf, images):
    parSize = conf['parallel']['parpoolSize']
    avg_res, avgrelu_res, sqrtvar_layer = getAvgResponse(images, conf, net)
    obj_num = images.shape[0]
    stat_all = dict(
        avg_res = avg_res,
        avgrelu_res = avgrelu_res,
        sqrtvar_layer = sqrtvar_layer,
        objNum = obj_num
    )

    batch_f = [[] for _ in range(obj_num)]
    batch_f_flip = [[] for _ in range(obj_num)]

    # stat = getStat(images)
    
    for id_start in range(0, obj_num, parSize):
        for par in range(0, min(obj_num-id_start, parSize)):
            obj_id = id_start+par
            # 10
            res = getObjFeatures(images[obj_id], net, 0)
            res_flip = getObjFeatures(images[obj_id], net, 0)

            for layer_id in range(conf['convnet']['targetLayers'].shape[0]):
                
                ori_layer = conf['convnet']['targetLayers'][layer_id]
                res[ori_layer] = getX(res, layer_id, conf, stat_all)
                res_flip[ori_layer] = getX(res_flip, layer_id, conf, stat_all)

            batch_f[obj_id] = roughCNNCompress(res, conf)
            batch_f_flip[obj_id] = roughCNNCompress(res, conf)
    
    return batch_f, batch_f_flip

def getX(res, layer_id, conf, stat_all):
    x = res[conf['convnet']['targetLayers'][layer_id]]
    tmp = x / stat_all['sqrtvar_layer']['x'][layer_id]
    tmp = np.where(tmp>0, tmp, -1)
    
    return tmp

def getAvgResponse(images, conf, net):
    obj_num = float(images.shape[0])
    # 3
    layer_num = conf['convnet']['targetLayers'].shape[0] # [7, 8, 9]
    # mean, std-->[3]
    # stat = getStat(images)
    
    avg_res = {'x': [[] for _ in range(layer_num)]}
    avgrelu_res = {'x': [[] for _ in range(layer_num)]}
    sqrtvar_layer = {'x': [[] for _ in range(layer_num)]}
    
    for i in tqdm.tqdm(range(int(obj_num)), unit='img'):
        res = getObjFeatures(images[i], net, 0)
        for layer in range(layer_num):
            x = res[conf['convnet']['targetLayers'][layer]]
            if i == 0:
                avg_res['x'][layer] = x
                avgrelu_res['x'][layer] = np.maximum(x, 0)
                sqrtvar_layer['x'][layer] = x**2
            else:
                avg_res['x'][layer] += x
                avgrelu_res['x'][layer] += np.maximum(x, 0)
                sqrtvar_layer['x'][layer] += x**2
    
    for layer in range(layer_num):
        avg_res['x'][layer] /= obj_num
        avgrelu_res['x'][layer] /= obj_num
        sqrtvar_layer['x'][layer] = np.sqrt(sqrtvar_layer['x'][layer] / obj_num - avg_res['x'][layer]**2)

    return avg_res, avgrelu_res, sqrtvar_layer

def roughCNNCompress(res, conf):
    num = len(res)
    res_c = {
        'size': [[] for _ in range(num)],
        'x': [[] for _ in range(num)],
        'rangeX': [[] for _ in range(num)],
        'minX': [[] for _ in range(num)]
    }
    for i in conf['convnet']['targetLayers']:
        # [c, h, w]
        x = res[i]
        # [h, w, c]
        the_size = np.array([x.shape[1], x.shape[2], x.shape[0]], dtype=np.uint16)
        # [h, w, c]
        x = x.transpose(1, 2, 0)
        res_c['size'][i] = the_size
        # [1, 1, c]
        min_x = x.min((0, 1), keepdims=True)
        x -= min_x
        # [1, 1, c]
        rangeX = x.max((0, 1), keepdims=True)
        res_c['minX'][i] = np.array(min_x, dtype=np.float32)
        res_c['rangeX'][i] = np.array(rangeX, dtype=np.float32)
        demo = np.zeros((1, 1, x.shape[-1]), dtype=np.float32)
        # feature maps中某些通道的map全为0，分母为0导致计算错误
        for j in range(rangeX.shape[-1]):
            tmp = rangeX[0, 0, j]
            if tmp != 0.0:
                demo[0, 0, j] = 255.0/tmp
            else:
                demo[0, 0, j] = 0.0

        res_c['x'][i] = np.array(x*demo, dtype=np.uint8)
    return res_c

def extractCNNFeatures():
    nets = {'unet':UNetEP, 'unetse':UNetSEEP,'attunet':AttUNet_EP, 'attunetwa':AttUNetEPWA}
    netname = 'unet'
    conf = getNetParameters(netname)

    dataname = 'BTCV2d_spleen'

    with open(f'{dataname}/images.pkl', 'rb') as f:
        img = pickle.load(f)
        # images[_neg]-->[num, h, w, c]
        images = img['images']
    
    net = nets[netname](1, 1).cuda()
    state_dict = torch.load(f'{dataname}/unet-btcv2dspleen-e70-b9442.pth')
    net.load_state_dict(state_dict)
    batch_f, batch_f_flip = getRoughCNN(net, conf, images)

    # matlab从1开始，python从0开始
    conf['convnet']['targetLayers'] += 1
    conf['convnet']['lastLayer'] += 1

    # 使用numpy==1.21.2版本不会报错，1.24.3会报错
    scio.savemat(f'{dataname}/{netname}/roughCNN.mat',
        {
            'batch_f': batch_f,
            'batch_f_flip': batch_f_flip,
            'conf': conf
        }
    )

if __name__ == '__main__':
    extractCNNFeatures()
    
