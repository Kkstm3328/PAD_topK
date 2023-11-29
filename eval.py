#利用他的eval重新写个
#使用格式同他的train
import os
import random
from copy import deepcopy

import cv2
import imageio
import numpy as np
import skimage
import torch
from torch.utils.data import DataLoader
import matplotlib as plt

import datasets.LEGO_3D as lego
from datasets.LEGO_3D import LEGODataset
from util.inerf_helpers import camera_transf
from sklearn.metrics import roc_auc_score
from util.nerf_helpers import load_nerf
from util.render_helpers import get_rays, render, to8b
from util.utils import (config_parser, find_POI,
                   img2mse, load_blender_ad, retrieval_loftr, AverageMeter, load_imgs_database)
import argparse
import importlib
import matplotlib
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology
from scipy.ndimage import gaussian_filter
import imageio
from sklearn import datasets
from util.metric import *
import numpy as np
import matplotlib.pyplot as plt
from util.model_helper import ModelHelper
from util.utils import *
from efficientnet_pytorch import EfficientNet
from easydict import EasyDict
import yaml
import logging
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1024
random.seed(seed)
torch.manual_seed(seed)

logger = logging.getLogger("global_logger")

parser = config_parser()
# parser.add_argument('--obj', type=str, default='01Gorilla')
parser.add_argument('--data_type', type=str, default='mvtec')
parser.add_argument('--dataset_path', type=str, default='./data/LEGO-3D')
parser.add_argument('--output_path',type=str,default='./output')
parser.add_argument('--checkpoint_dir', type=str, default='.')
parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
# parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--img_resize', type=int, default=128)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--resize', type=int, default=400)
args = parser.parse_args()

def update_config(config):      #照搬uniad
    # update feature size
    _, reconstruction_type = config.net[2].type.rsplit(".", 1)
    if reconstruction_type == "UniAD":
        input_size = config.dataset.input_size
        outstride = config.net[1].kwargs.outstrides[0]
        assert (
            input_size[0] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        assert (
            input_size[1] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        feature_size = [s // outstride for s in input_size]
        config.net[2].kwargs.feature_size = feature_size

    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config.net[0].kwargs.pop("outlayers")
        config.net[0].kwargs.outblocks = outblocks
    config.net[0].kwargs.outstrides = outstrides
    config.net[1].kwargs.outplanes = [sum(outplanes)]

    return config

def get_a_image_score(obs_img, rec_img, model):     #不过他这原本处理时400*400的图片干嘛转成224*224来算
    """
    #目前还不知道之后咋弄，就和一个单独的rec_img来计算了
    """
    import pdb; pdb.set_trace()
    obs_img = torch.tensor(obs_img.transpose((2, 0, 1))).unsqueeze(0)    # numpy:hwc -> tensor:1chw
    rec_img = torch.tensor(rec_img.transpose((2, 0, 1))).unsqueeze(0)
    MSE_loss = nn.MSELoss(reduction='none')
    
    obs_feature=model(obs_img)              #过特征提取器
    rec_feature=model(rec_img)
    # pred_mask=compare_feature(obs_feature,rec_feature)
    score = MSE_loss(obs_img, rec_img).sum(1, keepdim=True)     #image-level的loss      MSE_loss后为: 1*C*H*W, 再沿c求和
    for i in range(len(obs_feature)):           #对四层backbone得到的feature都计算loss, 所以总loss为img+4层feature的
        s_act = obs_feature[i]
        mse_loss = MSE_loss(s_act, rec_feature[i]).sum(1, keepdim=True)
        score += F.interpolate(mse_loss, size=224, mode='bilinear', align_corners=False)
    
    score = score.squeeze(1).cpu().numpy()
    for i in range(score.shape[0]):
        score[i] = gaussian_filter(score[i], sigma=4)
    return score

def cal_performance(scores, gt_mask_list, gt_list):
    scores = np.asarray(scores).squeeze()
    # normalization
    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score) 

    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

    import pdb; pdb.set_trace()
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)    #变成 n * (224 * 224)
    # img_scores = scores.reshape(scores.shape[0], -1).mean(axis=1)
    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    print('image ROCAUC: %.3f' % (img_roc_auc))

    save_result(args.class_name, per_pixel_rocauc, img_roc_auc)

def save_result(class_name, per_pixel_rocauc, img_roc_auc):
    res_path = "./result.csv"
    if not os.path.exists(res_path):
        df = pd.DataFrame(columns=['clsname', 'pixel ROCAUC', 'image ROCAUC'])
        df.to_csv(res_path, mode='w+', index=False)
    df = pd.read_csv(res_path)
    df.loc[len(df)] = [class_name, per_pixel_rocauc, img_roc_auc]
    df.to_csv(res_path, mode='w+', index=False)

def main():
    with open("retrieval/config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config = update_config(config)
    
    model = ModelHelper(config.net)     #backbone
    model.eval()
    model.cuda()

    dataset_path=args.dataset_path
    class_name=args.class_name
    output_path=args.output_path
    K = args.K
    resize = args.resize

    # 准备数据
    # imgs_database, _, _, imgs_path = load_blender_ad(         #得到检索的database，数据集集中的train
    #         args.data_dir, class_name, args.half_res, args.white_bkgd)
    imgs_database, imgs_path = load_imgs_database(args.data_dir, class_name, resize)

    lego_dataset = LEGODataset(dataset_path=dataset_path,      #测试集，实际上只含有类里面test的内容
                                   class_name=class_name,
                                   resize=resize)
    
    lego_loader = DataLoader(dataset=lego_dataset, 
                             batch_size= 1,
                             pin_memory=False)

    # 欠一个createmodel
    # model.eval()

    test_imgs = list()
    gt_mask_list = list()
    gt_list = list()

    # losses = AverageMeter(0)
    scores = list()

    with torch.no_grad():
        for i, (x, y, mask, xpath, _) in enumerate(lego_loader):  #mask和x： n*C*H*W, 未归一化
            print("{}/{}   queryimg is :{}".format(i+1, len(lego_loader.dataset), xpath))
            #一般 np是HWC， tensor是CHW

            # tmp = [itm for itm in mask.cpu().numpy().flatten() if (itm!=0) and (itm!=1)]
            # if len(tmp)>0:
            #     import pdb; pdb.set_trace()
        
            test_imgs.extend(x.cpu().numpy().squeeze(axis=0).transpose((1, 2, 0)))      # CHW -> HWC
            gt_list.extend(y.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy().squeeze(axis=0))
            obs_img = x.cpu().numpy().squeeze(axis=0).transpose((1, 2, 0))   #就是query
            
            #forward
            match_indexes = retrieval_loftr(imgs_database, obs_img, K)
            match_imgs = imgs_database[match_indexes]   # K * H * W * C

            # #######
            # # 康康match的img
            # match_paths = []
            # for index in match_indexes:
            #     match_paths.append(imgs_path[index])
            # print("query_img is: {}".format(xpath))
            # for match_path in match_paths:
            #     print("match img: {}".format(match_path))
            # import pdb; pdb.set_trace()
            # #######

            # 肯定要改
            # import pdb; pdb.set_trace()
            score = get_a_image_score(obs_img, match_imgs[0], model)
            scores.append(score)
            # cal_performance(scores, gt_mask_list, eval_resize)
            
        cal_performance(scores, gt_mask_list, gt_list)



    
    
    


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()