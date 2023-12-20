from copy import deepcopy
import importlib
import json
import os
from operator import index

import cv2
import imageio
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity
from util.model_helper import ModelHelper
from retrieval.loftr import LoFTR, default_cfg
from retrieval.retrieval import *
from easydict import EasyDict
import yaml

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import logging

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,        #!!!!!!!重要，这里指定了配置文件
                        help='config file path')
    parser.add_argument("--data_dir", type=str, default='./data/nerf_synthetic/',
                        help='path to data')
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts',
                        help='folder with saved checkpoints')
    parser.add_argument("--ckpt_name", type=str, 
                        help='name of ckpt')
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--retrieval_ans_dir", type=str, default="./retrieval_ans_dir/LEGO-3D",
                        help="save and load for LoFTR retrieval result")
    
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--dataset_path', type=str, default='./data/LEGO-3D')

    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--lrate", type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resize', type=int, default=400)

    # parser.add_argument('--img_resize', type=int, default=128)
    # parser.add_argument('--crop_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=None)

    #yes, is me! my options
    parser.add_argument("--feature_h", type=int, help="height of neck output features")
    parser.add_argument("--feature_w", type=int, help="width of neck output features")
    parser.add_argument("--neck_out_channel_dim", type=int, help="channel_dim of neck output features")

    parser.add_argument("--class_name", type=str, default='01Gorilla', help='LEGO-3D anomaly class')
    parser.add_argument("--K", type=int, default=1, help="retrival top-K pose similar image")
    parser.add_argument("--neighbor_size", type=int, help="neighbor for unmask")
    parser.add_argument("--hidden_dim", type=int, default=256, help="hidden_dim, channel_dim in TransformerEncoder")
    parser.add_argument("--num_encoder_layers", type=int, help="number of encoder_layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="dim_feedforward in TransformerEncoderLayer")


    return parser


def rot_psi(phi): return np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]])


def rot_theta(th): return np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]])


def rot_phi(psi): return np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])


def trans_t(t): return np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]])


def load_blender(data_dir, model_name, obs_img_num, half_res, white_bkgd, *kwargs):

    with open(os.path.join(data_dir + str(model_name) + "/obs_imgs/", 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    img_path = os.path.join(data_dir + str(model_name) +
                            "/obs_imgs/", frames[obs_img_num]['file_path'] + '.png')
    img_rgba = imageio.imread(img_path)
    # rgba image of type float32
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32)
    H, W = img_rgba.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if white_bkgd:
        img_rgb = img_rgba[..., :3] * \
            img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]
    imageio.imwrite("horse.png",img_rgb)
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    obs_img_pose = np.array(
        frames[obs_img_num]['transform_matrix']).astype(np.float32)
    phi, theta, psi, t = kwargs
    start_pose = trans_t(t) @ rot_phi(phi/180.*np.pi) @ rot_theta(theta /
                                                                  180.*np.pi) @ rot_psi(psi/180.*np.pi)  @ obs_img_pose
    # image of type uint8
    return img_rgb, [H, W, focal], start_pose, obs_img_pose


def load_blender_AD(data_dir, model_name, obs_img_num, half_res, white_bkgd, method,**kwargs):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms_train.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])+'.png'
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)

            imgs.append(img)
            poses.append(pose)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # imgs=np.array(imgs)
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)

    H, W = int(H), int(W)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    img_path = os.path.join(data_dir, str(model_name),'anomaly',str(obs_img_num)+".png")
    img_rgba = imageio.imread(img_path)
    # rgba image of type float32
    img_rgba = (np.array(img_rgba) / 255.).astype(np.float32)
    if white_bkgd and img_rgba.shape[-1]==4:
        img_rgb = img_rgba[..., :3] * \
            img_rgba[..., -1:] + (1. - img_rgba[..., -1:])
    else:
        img_rgb = img_rgba[..., :3]
    img_rgb = np.asarray(img_rgb*255, dtype=np.uint8)
    index_best = 0
    score_best = 0.5
    initial_pose = np.zeros([4, 4])
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(len(imgs)):
            imgs_half_res[i] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs=imgs_half_res
    
    # use lpips
    if method=='lpips':
        for i in range(len(imgs)):
            score=calculate_lpips(imgs[i],img_rgb,'vgg')
            if score < score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("lpips_min:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    elif method=='ssim':
    # use SSIM
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],img_rgb,'ssim')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("SSIM_max:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    # image of type uint8
    return img_rgb, [H, W, focal],initial_pose,score_best

def load_imgs_database(data_dir, cls_name, resize):
    """
    加载训练集中train内容供检索
    data_dir: 数据集path
    cls_name:
    ---------------------------------
    return:
        imgs<ndarray>
        imgs_path<list>
    """
    # import pdb; pdb.set_trace()
    imgs = list()
    train_path = os.path.join(data_dir, cls_name, "train", "good")
    imgs_path = sorted([os.path.join(train_path, f)for f in os.listdir(train_path)if f.endswith('.png')])
    tfms = transforms.Compose([
        transforms.Resize(resize), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    for img_path in imgs_path:
        img = Image.open(img_path).convert("RGB")
        img = tfms(img)
        imgs.append(img)
    stacked_tensor = torch.stack(imgs)
    imgs = stacked_tensor.numpy().transpose((0, 2, 3, 1))   # nCHW -> nHWC
    return imgs, imgs_path

def load_blender_ad(data_dir, model_name,  half_res, white_bkgd):
    # load train nerf images and poses
    meta = {}
    with open(os.path.join(data_dir, str(model_name), 'transforms.json'), 'r') as fp:
        meta["train"] = json.load(fp)
        imgs = []
        imgs_path = []
        poses = []
        for frame in meta["train"]['frames']:
            fname = os.path.join(data_dir, str(model_name), frame['file_path'])
            img = imageio.imread(fname)
            img = (np.array(img) / 255.).astype(np.float32)
            if white_bkgd and img.shape[-1]==4:
                img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
            else:
                img = img[..., :3]
            img = np.asarray(img*255, dtype=np.uint8)
            # img = tensorify(img)
            pose = (np.array(frame['transform_matrix']))
            pose = np.array(pose).astype(np.float32)
            imgs_path.append(fname)
            imgs.append(img)
            poses.append(pose)
            
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['train']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # imgs=np.array(imgs)
    imgs = np.stack(imgs, 0)
    poses = np.array(poses)
    H, W = int(H), int(W)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i in range(len(imgs)):
            imgs_half_res[i] = cv2.resize(imgs[i], (W, H), interpolation=cv2.INTER_AREA)
        imgs=imgs_half_res.astype(np.uint8)
    return imgs,[H, W, focal],poses, imgs_path

def find_nearest(imgs,obs_img,poses,method):
    # use lpips
    score_best=0.5
    if method=='lpips':
        for i in range(len(imgs)):
            score=calculate_lpips(imgs[i],obs_img,'vgg')
            if score < score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("lpips_min:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    elif method=='ssim':
    # use SSIM
        for i in range(len(imgs)):
            score,_=calculate_resmaps(imgs[i],obs_img,'ssim')
            if score > score_best:
                score_best = score
                index_best = i
                initial_pose = poses[i]
        print("SSIM_max:",score_best)
        print("index_best:",index_best)
        print("start_pose:",initial_pose)
    return initial_pose
        
def calculate_lpips(img1,img2,net='vgg',use_gpu=True):
    ## Initializing the model
    loss_fn = lpips.LPIPS(net)
    img1 = lpips.im2tensor(img1)  # RGB image from [-1,1]
    img2 = lpips.im2tensor(img2)

    if use_gpu:
        img1 = img1.cuda()
        img2 = img2.cuda()
    score = loss_fn.forward(img1, img2)
    return score

    
def resmaps_ssim(img_input,img_pred):
    score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
    return score,resmap

def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def resmaps_l1(imgs_input, imgs_pred):
    resmaps = np.abs(imgs_input - imgs_pred)
    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def calculate_resmaps(img_input, img_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if img_input.shape[-1] == 3:
        img_input_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_pred_gray = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    else:
        img_input_gray = img_input
        img_pred_gray = img_pred

    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(img_input_gray, img_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(img_input_gray, img_pred_gray)
    # if dtype == "uint8":
        # resmaps = img_as_ubyte(resmaps)
    return scores, resmaps
def rgb2bgr(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr
def bgr2rgb(img_bgr):
    img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    return img_rgb

def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imwrite(title, img_bgr)
    # cv2.imshow(title, img_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_POI(img_rgb, DEBUG=False):  # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected_points.png", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy  # pixel coordinates


# Misc
def img2mse(x, y): return torch.mean((x - y) ** 2)
def MAPE(x,y):return torch.mean(torch.abs((x-y)/y))
def Relative_L2(x,y):return torch.mean(torch.abs((x-y)**2/y**2))
def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)

# Load llff data

# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(
            len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    def p34_to_44(p): return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i,
                                [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]

    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, bds


def load_llff_data(data_dir, model_name, obs_img_num, *kwargs, factor=8, recenter=True, bd_factor=.75, spherify=False):
    # factor=8 downsamples original imgs by 8x
    poses, bds, imgs = _load_data(
        data_dir + str(model_name) + "/", factor=factor)
    print('Loaded', data_dir + str(model_name) + "/", bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, bds = spherify_poses(poses, bds)

    #images = images.astype(np.float32)
    images = np.asarray(images * 255, dtype=np.uint8)
    poses = poses.astype(np.float32)
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    obs_img = images[obs_img_num]
    obs_img_pose = np.concatenate(
        (poses[obs_img_num], np.array([[0, 0, 0, 1.]])), axis=0)
    phi, theta, psi, t = kwargs
    start_pose = rot_phi(phi/180.*np.pi) @ rot_theta(theta/180. *
                                                     np.pi) @ rot_psi(psi/180.*np.pi) @ trans_t(t) @ obs_img_pose
    return obs_img, hwf, start_pose, obs_img_pose, bds


def pose_retrieval(imgs,obs_img,poses):
    # Prepare model.
    model = load_model(pretrained_model='./retrieval/model/net_best.pth', use_gpu=True)

    # Extract database features.
    gallery_feature = extract_feature(model=model, imgs=imgs)

    # Query.
    query_image = transform_query_image(obs_img)

    # Extract query features.
    query_feature = extract_feature_query(model=model, img=query_image)

    # Sort.
    similarity, index = sort_img(query_feature, gallery_feature)

    return poses[index[0]]


def pose_retrieval_efficient(imgs,obs_img,poses):
    # Prepare model.
    model = load_model_efficient()

    # Extract database features.
    gallery_feature = extract_feature_efficient(model=model, imgs=imgs)

    # Extract query features.
    query_feature = extract_feature_query_efficient(model=model, img=obs_img)

    # Sort.
    similarity, index = sort_img_efficient(query_feature, gallery_feature)

    return poses[index]

def retrieval_loftr(imgs, obs_img, K):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("retrieval/model/indoor_ds_new.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    match_nums = []
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        match_nums.append(len(mconf))
        # match_num=len(mconf) ####
        # if match_num>max_match:
        #     max_match=match_num
        #     max_index=i
    # import pdb; pdb.set_trace()
    match_topK_indexes = np.array(match_nums).argsort()[-K:][::-1]
    return match_topK_indexes

class AverageMeter(object):     #UniAD的，好用
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):   #val为loss值
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]     #取history最后个元素
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def anomaly_score_map(score, path):
    plt.close('all')
    # import pdb; pdb.set_trace()
    new_path = path[0]
    # 使用 os.path 来获取文件路径的各个部分
    directory, filename = os.path.split(new_path)
    # 分割目录并找到以 'LEGO-3D' 开始的部分
    directory_parts = directory.split(os.path.sep)
    lego_3d_start = next(part for part in directory_parts if part.startswith('LEGO-3D'))

    new_path = os.path.join(directory_parts[0], 'score_map', directory_parts[2], directory_parts[3], directory_parts[4], directory_parts[5], filename)
    new_dir = os.path.dirname(new_path)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # 假设 anomaly_scores 是一个 224x224 的 PyTorch 张量，表示异常得分
    # 示例数据，实际应用中替换为您的张量数据
    anomaly_scores = torch.tensor(score).squeeze(0)

    # 创建热力图
    plt.figure(figsize=(8, 8))
    plt.imshow(anomaly_scores.cpu().numpy(), cmap='hot', interpolation='nearest')

    # 添加颜色条
    plt.colorbar()

    # 添加标签和标题
    plt.title('异常得分热力图')
    plt.xlabel('列编号')
    plt.ylabel('行编号')

    # 保存热力图为图片文件（可以根据需要选择不同的文件格式，例如 PNG、JPEG、PDF 等）
    plt.savefig(new_path)

    # 显示热力图
    # plt.show()


def preRetrieval(dataloader, imgs_database, retrieval_ans_dir, class_name, save_path, K=15):
    """
        离线检索对应imgs
        存储<xpath, ndarray[K]>
        ndarray存储K个imgs_database的下标
    """
    matched_dics = {}
    # import pdb; pdb.set_trace()
    for i, (x, xpath, *_) in enumerate(dataloader):
        # x: b c h w
        print(x.device)
        cur_batch_size = x.shape[0]
        for j in range(cur_batch_size):
            # if xpath[j] == "./data/LEGO-3D/01Gorilla/test/Stains/21.png":
            #     import pdb; pdb.set_trace()
            obs_img = x[j].cpu().numpy().transpose((1, 2, 0))   # h w c
            match_indexes = retrieval_loftr(imgs_database, obs_img, K)  #ndarray[K]
            matched_dics[xpath[j]] = match_indexes
            print("preRetrievaling: {}\t{}/{}".format(xpath[j], (i*cur_batch_size)+j+1, len(dataloader)*cur_batch_size))
    
    if not os.path.exists(retrieval_ans_dir):
        os.makedirs(retrieval_ans_dir)
    with open(save_path, 'w') as file:
        for xpath, np_array in matched_dics.items():
            np_array_str = ','.join(map(str, np_array))  # 将 np_array 转换为逗号分隔的字符串
            file.write(f"{xpath}:{np_array_str}\n")

def load_Retrieval_dics(path):
    """
        return dics: <xpath, ndarray[K]>
    """
    matched_dics = {}
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            np_array = np.array(list(map(int, value.split(','))))  # 将字符串转换回 numpy 数组
            matched_dics[key] = np_array
    return matched_dics

def cal_performance(preds, masks, labels):
    preds = torch.stack(preds).cpu().numpy()    # n 1 h w
    masks = torch.stack(masks).cpu().numpy()
    labels = [item.cpu().numpy() for item in labels]

    # 将预测的图像列表和 ground truth 的 mask 列表展平
    pred_array = preds.flatten()
    mask_array = masks.flatten()

    # 计算像素级 AUROC
    pixel_auroc = roc_auc_score(mask_array, pred_array)

    image_scores = [np.max(pred[0]) for pred in preds]
    # import pdb; pdb.set_trace()
    image_auroc = roc_auc_score(labels, image_scores)

    formatted_labels = f'{len(labels)} items: {labels}'
    formatted_img_scores = f'{len(image_scores)} items: {image_scores}'
    logging.info(
        "labels:\t\t:{0}".format(formatted_labels)
    )
    logging.info(
        "img_score:\t{0}".format(formatted_img_scores)
    )
    return pixel_auroc, image_auroc