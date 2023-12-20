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
from util.utils import (config_parser, retrieval_loftr, AverageMeter, load_imgs_database, preRetrieval, load_Retrieval_dics, cal_performance)
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
import pandas
from models.recons import fuT
from datetime import datetime
from einops import rearrange

pred_dir = "./preds/LEGO-3D/big"
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 1024
random.seed(seed)
torch.manual_seed(seed)

parser = config_parser()
args = parser.parse_args()

# log
log_dir = args.log_dir
log_dir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d'))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, datetime.now().strftime('%H-%M-%S') + '.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO)

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

def main():
    dataset_path=args.dataset_path
    class_name=args.class_name
    ckpt_dir = args.ckpt_dir
    ckpt_name = args.ckpt_name
    retrieval_ans_dir = args.retrieval_ans_dir

    epoch = args.epoch
    lrate = args.lrate
    batch_size = args.batch_size
    resize = args.resize

    feature_size = [args.feature_h, args.feature_w]
    neck_out_channel_dim = args.neck_out_channel_dim

    K = args.K
    neighbor_size = args.neighbor_size
    hidden_dim = args.hidden_dim
    num_encoder_layers = args.num_encoder_layers
    dim_feedforward = args.dim_feedforward


    # 准备数据
    imgs_database, imgs_path = load_imgs_database(args.data_dir, class_name, resize)    # np [n, h, w, c]

    lego_dataset = LEGODataset(dataset_path=dataset_path,
                                   class_name=class_name,
                                   resize=resize,
                                   isTrainingSet=True)
    
    train_loader = DataLoader(dataset=lego_dataset, 
                             batch_size=batch_size,
                             pin_memory=False)

    val_dataset = LEGODataset(dataset_path=dataset_path,
                                   class_name=class_name,
                                   resize=resize,
                                   isTrainingSet=False)
    val_loader = DataLoader(dataset=val_dataset, 
                             batch_size=batch_size,
                             num_workers=0,
                             pin_memory=False)
    

    # preRetrieval by LoFTR
        # train
    train_retrieval_ans_file_path = os.path.join(retrieval_ans_dir, class_name+'_train.txt')
    if not os.path.exists(train_retrieval_ans_file_path):
        preRetrieval(train_loader, imgs_database, retrieval_ans_dir, class_name, train_retrieval_ans_file_path)
    train_matched_dics = load_Retrieval_dics(train_retrieval_ans_file_path)
        # validate
    val_retrieval_ans_file_path = os.path.join(retrieval_ans_dir, class_name+'_val.txt')
    if not os.path.exists(val_retrieval_ans_file_path):
        preRetrieval(val_loader, imgs_database, retrieval_ans_dir, class_name, val_retrieval_ans_file_path)
    val_matched_dics = load_Retrieval_dics(val_retrieval_ans_file_path)
    # val_matched_dics = None

    # best metric
    best_pixel_auroc = float("-inf")

    # create models
    with open("retrieval/config.yaml") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config = update_config(config)
    backboneNeck = ModelHelper(config.net)  #backbone and neck
    backboneNeck.cuda()

    model = fuT(hidden_dim, batch_size, feature_size, [resize, resize], K, neck_out_channel_dim, neighbor_size, num_encoder_layers, dim_feedforward)
    model.cuda()

    # load_model
    ckpt_path = os.path.join(ckpt_dir, ckpt_name+".pth.tar")
    ckpt = torch.load(ckpt_path)
    backboneNeck.load_state_dict(ckpt["backboneNeck_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])

    # optim
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lrate)   # lr固定先

    # loss
    mse_loss = nn.MSELoss()

    for cur_epoch in range(1):
        # train_one_epoch(
        #     class_name,
        #     train_loader, 
        #     backboneNeck, 
        #     model, 
        #     optimizer, 
        #     mse_loss, 
        #     train_matched_dics, 
        #     imgs_database, 
        #     K, 
        #     cur_epoch, 
        #     epoch, 
        #     batch_size
        # )

        # validate<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # if (cur_epoch+1) % 10 == 0:
        #     validate()
        avg_loss, pixel_auroc, image_auroc = validate(
            class_name,
            val_loader,
            backboneNeck,
            model,
            optimizer,
            mse_loss,
            val_matched_dics,
            imgs_database,
            K,
            cur_epoch, 
            epoch, 
            batch_size
        )

        print(
            "---->\t"
            "Class: {class_name}\t"
            "Epoch: [{cur_epoch}/{max_epoch}]\t"
            "avg_loss: {avg_loss:.5f}\t"
            "pix_auroc: {pix_auroc:.3f}\t"
            "img_auroc: {img_auroc:.3f}\t".format(
                class_name = class_name,
                cur_epoch = cur_epoch+1, max_epoch = epoch,
                avg_loss = avg_loss,
                pix_auroc = pixel_auroc,
                img_auroc = image_auroc
            )
        )

        logging.info(
            "---->\t"
            "Class: {class_name}\t"
            "Epoch: [{cur_epoch}/{max_epoch}]\t"
            "avg_loss: {avg_loss:.5f}\t"
            "pix_auroc: {pix_auroc:.3f}\t"
            "img_auroc: {img_auroc:.3f}\t".format(
                class_name = class_name,
                cur_epoch = cur_epoch+1, max_epoch = epoch,
                avg_loss = avg_loss,
                pix_auroc = pixel_auroc,
                img_auroc = image_auroc
            )
        )
        # if is best
        #   save model<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # if pixel_auroc > best_pixel_auroc:
        #     print("saving best model......")
        #     torch.save(
        #         {
        #             "epoch": cur_epoch+1,
        #             "best_metric": {
        #                 "pixel_auroc": pixel_auroc, 
        #                 "image_auroc": image_auroc,
        #                 },
        #             "backboneNeck_state_dict": backboneNeck.state_dict(),
        #             "model_state_dict": model.state_dict(),
        #             "optimizer": optimizer.state_dict(),
        #         },
        #         os.path.join(ckpt_dir, ckpt_name+".pth.tar")
        #     )

def train_one_epoch(
        class_name,
        train_loader,
        backboneNeck,
        model,
        optimizer,
        criterion,
        matched_dics,
        imgs_database,
        K,
        cur_epoch,
        max_epoch,
        batch_size
):
    """
        matched_dics: dicts of <img_path, [indexes of imgs_database]>
    """
    backboneNeck.eval()  
    for param in backboneNeck.parameters():
        param.requires_grad = False

    model.train()

    for i, (x, xpath) in enumerate(train_loader):
        # obs_img = x.cpu().numpy().squeeze(axis=0).transpose((1, 2, 0))

        obs_img = x.to(device)     # b c h w

        tmp_imgs_list= []
        for cur_xpath in xpath: #for each batch
            matched_indexes = matched_dics[cur_xpath][0:K]   # np [0:K]
            matched_imgs = imgs_database[matched_indexes]   # np shape==[K, h, w, c]
            tmp_imgs_list.append(matched_imgs)
        matched_imgs_K = rearrange(torch.tensor(np.stack(tmp_imgs_list)).to(device), "b k h w c -> k b c h w")   #  K * b * c * h * w 

        # forward
        # import pdb; pdb.set_trace()
        obs_feature = backboneNeck(obs_img)

        matched_feature_list_K = list()
        for j in range(K):
            feature = backboneNeck(matched_imgs_K[j])
            matched_feature_list_K.append(feature)
        
        rec_feature, _ = model(obs_feature, matched_feature_list_K)

        # loss
        loss = criterion(obs_feature, rec_feature)

        # optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i+1) % 10 == 0:
        #     print(f'Epoch [{cur_epoch+1}/{max_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print(
            "training: "
            "Class: {class_name}\t"
            "Time: {0}\t"
            "Epoch: [{1}/{2}]\t"
            "Step: [{3}/{4}]\t"
            "Loss: {loss:.5f}\t".format(
                datetime.now().strftime('%H:%M:%S'),
                cur_epoch+1, max_epoch,
                i+1, len(train_loader),
                loss = loss,
                class_name = class_name
            )
        )
        logging.info(
            "training: "
            "Class: {class_name}\t"
            "Time: {0}\t"
            "Epoch: [{1}/{2}]\t"
            "Step: [{3}/{4}]\t"
            "Loss: {loss:.5f}\t".format(
                datetime.now().strftime('%H:%M:%S'),
                cur_epoch+1, max_epoch,
                i+1, len(train_loader),
                loss = loss,
                class_name = class_name
            )
        )

def validate(
        class_name,
        val_loader,
        backboneNeck,
        model,
        optimizer,
        criterion,
        matched_dics,
        imgs_database,
        K,
        cur_epoch,
        max_epoch,
        batch_size
):
    backboneNeck.eval()  
    for param in backboneNeck.parameters():
        param.requires_grad = False

    model.eval()

    # record
    total_loss = 0.0
    total_samples = 0
    preds_list = []
    pred_paths_list = []
    gts_list = []
    gt_masks_list = []
    gt_mask_paths_list = []

    with torch.no_grad():
        for i, (x, xpath, gt, gt_mask, gt_mask_path) in enumerate(val_loader):
            # import pdb; pdb.set_trace()
            cur_batch_size = x.shape[0]

            obs_img = x     # b c h w

            tmp_imgs_list= []
            for cur_xpath in xpath: #for each batch
                matched_indexes = matched_dics[cur_xpath][0:K]   # np [0:K]
                matched_imgs = imgs_database[matched_indexes]   # np shape==[K, h, w, c]
                tmp_imgs_list.append(matched_imgs)
            matched_imgs_K = rearrange(torch.tensor(np.stack(tmp_imgs_list)).to(device), "b k h w c -> k b c h w")   #  K * b * c * h * w 

            # forward
            # import pdb; pdb.set_trace()
            obs_feature = backboneNeck(obs_img)

            matched_feature_list_K = list()
            for j in range(K):
                feature = backboneNeck(matched_imgs_K[j])
                matched_feature_list_K.append(feature)
            
            rec_feature, pred = model(obs_feature, matched_feature_list_K)  # b c h w       

            # loss
            loss = criterion(obs_feature, rec_feature)
            total_loss += loss * cur_batch_size   # current batch_size
            total_samples += cur_batch_size

            # record
            for j in range(cur_batch_size):
                preds_list.append(pred[j])
                pred_paths_list.append(xpath[j])
                gts_list.append(gt[j])
                gt_masks_list.append(gt_mask[j])    # gt_mask.shape == b 1 h w
                gt_mask_paths_list.append(gt_mask_path[j])

                # save_preds
                import pdb; pdb.set_trace()
                pred_path = os.path.join(pred_dir, class_name, '/'.join(xpath[j].split('/')[-3:]))
                tmp_dir = os.path.dirname(pred_path)
                os.makedirs(tmp_dir, exist_ok=True)
                pred_img = (pred[j].cpu().numpy() * 255).astype('uint8').squeeze()
                plt.imshow(pred_img, cmap='hot', interpolation='nearest')
                plt.savefig(pred_path)

            print(
                "\tshowPreds: "
                "Class: {class_name}\t"
                "Time: {0}\t"
                "Epoch: [{1}/{2}]\t"
                "Step: [{3}/{4}]\t"
                "Loss: {loss:.5f}\t".format(
                    datetime.now().strftime('%H:%M:%S'),
                    cur_epoch+1, 1,
                    i+1, len(val_loader),
                    loss = loss,
                    class_name = class_name
                )
            )

    
        avg_loss = total_loss / total_samples
    
        # cal performance
        pixel_auroc, image_auroc = cal_performance(preds_list, gt_masks_list, gts_list)
    return avg_loss, pixel_auroc, image_auroc

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()