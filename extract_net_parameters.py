from models import AttUNet_EP, UNetEP, VGG16_EP, AttUNetEPWA, UNetSEEP
import numpy as np
import torch.nn as nn

def getConvNet_VGG16(conf, net):
    conv_layers = []
    for i, tmp in enumerate(net):
        if isinstance(tmp, nn.Conv2d):
            conv_layers.append(i)
    conv_layers = np.array(conv_layers)
    valid_layers = np.array([8, 9, 11, 12])
    img_size = [224, 224]
    convnet = dict(convLayers = conv_layers, \
                   validLayers = valid_layers, imgSize = img_size)
    convnet = getConvNetPara(convnet, net)
    position_cand_num = 6
    pattern_density = np.array([0.05, 0.05, 0.1, 0.1])

    max_range = np.array([0.3, 0.3, 0.3, 0.3])
    deform_ratio = 3
    search_ = dict(
        maxRange = max_range,
        deform_ratio = deform_ratio
    )

    init_delta = 0.15
    max_delta = 1.0 / np.sqrt(pattern_density * (np.array([28, 28, 14, 14]) ** 2))
    min_delta = max_delta / 10
    deform_ = dict(
        init_delta = init_delta,
        max_delta = max_delta,
        min_delta = min_delta
    )

    map_delta = 0.025
    top_n = np.array([15, 15, 15, 0])
    top_m = np.array([600, 600, 600, 0])
    valid_tau = 0.1

    learn = dict(
        positionCandNum = position_cand_num,
        patternDensity = pattern_density,
        search_ = search_,
        deform_ = deform_,
        map_delta = map_delta,
        topN = top_n,
        topM = top_m,
        validTau = valid_tau
    )

    conf['convnet'] = convnet
    conf['learn'] = learn

    return conf

def getConvNet_UNet(conf, net):
    conv_layers = []
    for i, tmp in enumerate(net):
        if isinstance(tmp, nn.Conv2d):
            conv_layers.append(i)
    conv_layers = np.array(conv_layers)
    
    valid_layers = np.array([7, 8, 9])
    img_size = [256, 256]
    convnet = dict(convLayers = conv_layers, \
                   validLayers = valid_layers, imgSize = img_size)
    convnet = getConvNetPara(convnet, net)
    position_cand_num = 6.0
    pattern_density = np.array([0.05, 0.1, 0.1])

    max_range = np.array([0.3, 0.3, 0.3])
    deform_ratio = 3.0
    search_ = dict(
        maxRange = max_range,
        deform_ratio = deform_ratio
    )

    init_delta = 0.15
    max_delta = 1.0 / np.sqrt(pattern_density * (np.array([32, 16, 16]) ** 2))
    min_delta = max_delta / 10
    deform_ = dict(
        init_delta = init_delta,
        max_delta = max_delta,
        min_delta = min_delta
    )

    map_delta = 0.025
    top_n = np.array([15.0, 15.0, 0.0])
    top_m = np.array([600.0, 600.0, 0.0])
    valid_tau = 0.1

    learn = dict(
        positionCandNum = position_cand_num,
        patternDensity = pattern_density,
        search_ = search_,
        deform_ = deform_,
        map_delta = map_delta,
        topN = top_n,
        topM = top_m,
        validTau = valid_tau
    )

    conf['convnet'] = convnet
    conf['learn'] = learn

    return conf

def getConvNet_UNetSE(conf, net):
    conv_layers = []
    for i, tmp in enumerate(net):
        if isinstance(tmp, nn.Conv2d):
            conv_layers.append(i)
    conv_layers = np.array(conv_layers)
    
    valid_layers = np.array([7, 8, 9])
    img_size = [256, 256]
    convnet = dict(convLayers = conv_layers, \
                   validLayers = valid_layers, imgSize = img_size)
    convnet = getConvNetPara(convnet, net)
    position_cand_num = 6.0
    pattern_density = np.array([0.05, 0.1, 0.1])

    max_range = np.array([0.3, 0.3, 0.3])
    deform_ratio = 3.0
    search_ = dict(
        maxRange = max_range,
        deform_ratio = deform_ratio
    )

    init_delta = 0.15
    max_delta = 1.0 / np.sqrt(pattern_density * (np.array([32, 16, 16]) ** 2))
    min_delta = max_delta / 10
    deform_ = dict(
        init_delta = init_delta,
        max_delta = max_delta,
        min_delta = min_delta
    )

    map_delta = 0.025
    top_n = np.array([15.0, 15.0, 0.0])
    top_m = np.array([600.0, 600.0, 0.0])
    valid_tau = 0.1

    learn = dict(
        positionCandNum = position_cand_num,
        patternDensity = pattern_density,
        search_ = search_,
        deform_ = deform_,
        map_delta = map_delta,
        topN = top_n,
        topM = top_m,
        validTau = valid_tau
    )

    conf['convnet'] = convnet
    conf['learn'] = learn

    return conf

def getConvNet_AttUNet(conf, net):
    conv_layers = []
    for i, tmp in enumerate(net):
        if isinstance(tmp, nn.Conv2d):
            conv_layers.append(i)
    conv_layers = np.array(conv_layers)
    # 取最后三个卷积层生成解释图
    valid_layers = np.array([7, 8, 9])
    img_size = [256, 256]
    convnet = dict(convLayers = conv_layers, \
                   validLayers = valid_layers, imgSize = img_size)
    convnet = getConvNetPara(convnet, net)
    position_cand_num = 6.0
    pattern_density = np.array([0.05, 0.1, 0.1])

    max_range = np.array([0.3, 0.3, 0.3])
    deform_ratio = 3.0
    search_ = dict(
        maxRange = max_range,
        deform_ratio = deform_ratio
    )

    init_delta = 0.15
    max_delta = 1.0 / np.sqrt(pattern_density * (np.array([32, 16, 16]) ** 2))
    min_delta = max_delta / 10.0
    deform_ = dict(
        init_delta = init_delta,
        max_delta = max_delta,
        min_delta = min_delta
    )

    map_delta = 0.025
    top_n = np.array([15.0, 15.0, 0.0])
    top_m = np.array([600.0, 600.0, 0.0])
    valid_tau = 0.1

    learn = dict(
        positionCandNum = position_cand_num,
        patternDensity = pattern_density,
        search_ = search_,
        deform_ = deform_,
        map_delta = map_delta,
        topN = top_n,
        topM = top_m,
        validTau = valid_tau
    )

    conf['convnet'] = convnet
    conf['learn'] = learn

    return conf

def getConvNet_AttUNet_WithAtt(conf, net):
    conv_layers = []
    for i, tmp in enumerate(net):
        if isinstance(tmp, nn.Conv2d):
            conv_layers.append(i)
    conv_layers = np.array(conv_layers)
    # 取2个卷积层生成解释图
    valid_layers = np.array([5, 7])
    img_size = [256, 256]
    convnet = dict(convLayers = conv_layers, \
                   validLayers = valid_layers, imgSize = img_size)
    convnet = getConvNetPara(convnet, net)
    position_cand_num = 6.0
    pattern_density = np.array([0.05, 0.1])

    max_range = np.array([0.3, 0.3])
    deform_ratio = 3.0
    search_ = dict(
        maxRange = max_range,
        deform_ratio = deform_ratio
    )

    init_delta = 0.15
    max_delta = 1.0 / np.sqrt(pattern_density * (np.array([64, 32]) ** 2))
    min_delta = max_delta / 10
    deform_ = dict(
        init_delta = init_delta,
        max_delta = max_delta,
        min_delta = min_delta
    )

    map_delta = 0.025
    top_n = np.array([15.0, 0.0])
    top_m = np.array([600.0, 0.0])
    valid_tau = 0.1

    learn = dict(
        positionCandNum = position_cand_num,
        patternDensity = pattern_density,
        search_ = search_,
        deform_ = deform_,
        map_delta = map_delta,
        topN = top_n,
        topM = top_m,
        validTau = valid_tau
    )

    conf['convnet'] = convnet
    conf['learn'] = learn

    return conf

def getConvNetPara(convnet, net):
    # 10
    conv_num = len(convnet['convLayers'])

    last_layer_n = len(net)-1
    target_scale = np.zeros(conv_num)
    target_stride = np.zeros(conv_num)
    target_center = np.zeros(conv_num)
    for i in range(conv_num):
        layer_n = convnet['convLayers'][i]

        layer = net[layer_n]
        pad = layer.padding[0]
        scale = layer.kernel_size[0]
        stride = layer.stride[0]
        if i == 0:
            target_stride[i] = stride
            target_scale[i] = scale
            target_center[i] = (1+scale-pad*2)/2
        else:
            is_pool = 0
            pool_stride = 0
            pool_size = 0
            pool_pad = 0
            for j in range(convnet['convLayers'][i-1]+1, layer_n):
                if isinstance(net[j], nn.MaxPool2d):
                    is_pool = 1
                    pool_size = net[j].kernel_size
                    pool_stride = net[j].stride
                    pool_pad = net[j].padding
            target_stride[i] = (1+is_pool*(pool_stride-1))*stride*target_stride[i-1]
            target_scale[i] = target_scale[i-1]+is_pool*(pool_size-1)*target_stride[i-1]+target_stride[i]*(scale-1)
            if is_pool == 1:
                target_center[i] = (scale-pad*2-1)*pool_stride*target_stride[i-1]/2+(target_center[i-1]+target_stride[i-1]*(pool_size-2*pool_pad-1)/2)
            else:
                target_center[i] = (scale-pad*2-1)*target_stride[i-1]/2+target_center[i-1]

    convnet['targetScale'] = target_scale[convnet['validLayers']]
    convnet['targetStride'] = target_stride[convnet['validLayers']]
    convnet['targetCenter'] = target_center[convnet['validLayers']]
    convnet['targetLayers'] = convnet['convLayers'][convnet['validLayers']]
    convnet['lastLayer'] = last_layer_n

    convnet.pop('convLayers')
    convnet.pop('validLayers')

    return convnet

def getNetParameters(_type: str):
    conf = {}
    conf['parallel'] = {'parpoolSize':8}
    if _type.lower() == 'unet':
        model = UNetEP(1, 1)
        conf = getConvNet_UNet(conf, model.getLayers())
    elif _type.lower() == 'unetse':
        model = UNetSEEP(1, 1)
        conf = getConvNet_UNetSE(conf, model.getLayers())
    elif _type.lower() == 'vgg16':
        model = VGG16_EP(3, 2)
        conf = getConvNet_VGG16(conf, model.getLayers())
    elif _type.lower() == 'attunet':
        model = AttUNet_EP(1, 1)
        conf = getConvNet_AttUNet(conf, model.getLayers())
    elif _type.lower() == 'attunetwa':
        model = AttUNetEPWA(1, 1)
        conf = getConvNet_AttUNet_WithAtt(conf, model.getLayers())
    return conf