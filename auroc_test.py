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

def plot_fig(test_img, recon_imgs, scores, gts, threshold, save_dir,class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        recon_img = recon_imgs[i]
        recon_img = denormalization(recon_img)
        gt = gts[i]
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 6, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(recon_img)
        ax_img[1].title.set_text('Reconst')
        ax_img[2].imshow(gt, cmap='gray')
        ax_img[2].title.set_text('GroundTruth')
        ax = ax_img[3].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[3].imshow(img, cmap='gray', interpolation='none')
        ax_img[3].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[3].title.set_text('Predicted heat map')
        ax_img[4].imshow(mask, cmap='gray')
        ax_img[4].title.set_text('Predicted mask')
        ax_img[5].imshow(vis_img)
        ax_img[5].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}_png'.format(i)), dpi=100)
        plt.close()
        
def update_config(config):      #直接搬的UniAD
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

import pdb; pdb.set_trace()
with open("retrieval/config.yaml") as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
config = update_config(config)
model = ModelHelper(config.net)     #backbone
model.eval()
model.cuda()


def compare_feature(ref_feature,rgb_feature):
    loss=(ref_feature-rgb_feature)**2 
    result_sum=torch.mean(loss,axis=0) 
    result_sum = gaussian_filter(result_sum, sigma=4)
    return result_sum

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--obj', type=str, default='01Gorilla')
parser.add_argument('--data_type', type=str, default='mvtec')
parser.add_argument('--dataset_path', type=str, default='./data/LEGO-3D')
parser.add_argument('--output_path',type=str,default='./output')
parser.add_argument('--checkpoint_dir', type=str, default='.')
parser.add_argument("--grayscale", action='store_true', help='color or grayscale input image')
parser.add_argument('--batch_size', type=int, default=1)
# parser.add_argument('--img_resize', type=int, default=128)
parser.add_argument('--img_resize', type=int, default=224)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument("--K", type=int, default=1, help="retrival top-K pose similar image")
args = parser.parse_args()
dataset_path=args.dataset_path
class_name=args.obj
output_path=args.output_path

# 一大堆准备测试数据
fig, ax = plt.subplots(1, 2, figsize=(20, 10))
# fig_img_rocauc = ax[0]      #建两张图,应该没啥用
# fig_pixel_rocauc = ax[1]
img_size = 224
x, y, mask = [], [], []     #测试集img路径，对应label，作为gt的mask路径
transform_mask = transforms.Compose([transforms.Resize(img_size, Image.NEAREST),    #transforms.Compose()把参数（一个列表）的操作依次实现，
            transforms.ToTensor()])                 #所以transform_mask为一个函数
img_dir = os.path.join(dataset_path, class_name, 'test')    #测试集path
gt_dir = os.path.join(                                  #gt的path
    dataset_path, class_name, 'ground_truth')
output_dir=os.path.join(output_path,class_name)         #ouput的path，所以output为图？？？？？？？？
output_types=os.listdir(output_dir)                 #就是测试集的每个图像们，包括三种异常和good的图像们
output_types.sort(key=lambda x:int(x.split('_')[1]))
img_types = sorted(os.listdir(img_dir))     #测试集里面的不同类，三种异常+good
for img_type in img_types:      #得到前面x, y, mask
    img_type_dir = os.path.join(img_dir, img_type)      #   ./data/LEGO-3D/01Gorilla/test/Burrs 这种
    if not os.path.isdir(img_type_dir):
        continue
    img_fpath_list = sorted([os.path.join(img_type_dir, f)      #里面所有 PNG 文件的path，顺序为文件名
                            for f in os.listdir(img_type_dir)
                            if f.endswith('.png')])
    x.extend(img_fpath_list) # test path

    if img_type == 'good':
        y.extend([0] * len(img_fpath_list))     #good就往y里面增多个0
        mask.extend([None] * len(img_fpath_list))   #mask增多个None
    else:
        y.extend([1] * len(img_fpath_list))     #不good的label增1
        gt_type_dir = os.path.join(gt_dir, img_type)    # 三种异常之一的gt的path
        img_fname_list = [os.path.splitext(os.path.basename(f))[    # '0', '1' 这种不带png的
            0] for f in img_fpath_list]
        gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                            for img_fname in img_fname_list]
        mask.extend(gt_fpath_list)
test_imgs = list()
gt_mask_list = list()
gt_list = y
score_map_list=list()
scores=list()
pred_list=list()
recon_imgs=list()
gt_list_array=np.array(gt_list)
    # 准备gt_mask_list: gt的矩阵序列
for i in range(len(x)):
    if y[i] == 0:   #label=='good'
        gt_mask = torch.zeros([1,224,224])      #'good'的mask全0
    else:
        gt_mask = Image.open(mask[i])       # 这两步从路径得到mask矩阵
        gt_mask = transform_mask(gt_mask)
    # gt_mask=(gt_mask/255.0).astype(torch.uint8)

    gt_mask_list.extend(gt_mask.cpu().numpy())
    

front=gt_list.index(0)
end=len(gt_list)
gt_array=np.arange(front,end)   #测试集中good类的下标们
# gt_list_2=[1]*len(gt_list)

MSE_loss = nn.MSELoss(reduction='none')     #loss
tfms = transforms.Compose([         #又一个图像变换：大小resize + 变tensor + Normalization
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #三个通道的均值和标准差，这几个是好多数据集通用的，不用管
    ])
with torch.no_grad():
    for i,anomaly_type in enumerate(output_types):
        ref_path=os.path.join(output_dir,anomaly_type,'ref.png')    #训练集作query的img的path
        rgb_path=os.path.join(output_dir,anomaly_type,'rgb8.png')   #训练集里面重建的img的path
        ref=tfms(Image.open(ref_path).convert('RGB')).unsqueeze(0).cuda()
        rgb=tfms(Image.open(rgb_path).convert('RGB')).unsqueeze(0).cuda()
        ref_feature=model(ref)
        rgb_feature=model(rgb)
        # pred_mask=compare_feature(ref_feature,rgb_feature)
        score = MSE_loss(ref, rgb).sum(1, keepdim=True)     #image-level的loss
        for i in range(len(ref_feature)):           #对四层backbone得到的feature都计算loss, 所以总loss为img+4层feature的
            s_act = ref_feature[i]
            mse_loss = MSE_loss(s_act, rgb_feature[i]).sum(1, keepdim=True)
            score += F.interpolate(mse_loss, size=224, mode='bilinear', align_corners=False)
        
        score = score.squeeze(1).cpu().numpy()
        for i in range(score.shape[0]):
            score[i] = gaussian_filter(score[i], sigma=4)
        recon_imgs.extend(rgb.cpu().numpy())    #就直接把train阶段重建的img放进去了
        test_imgs.extend(ref.cpu().numpy())     #同上
        scores.append(score)

#计算指标
scores = np.asarray(scores).squeeze()       #把(n个)  n*1*224*224的np,展开成n*224*224
max_anomaly_score = scores.max()
min_anomaly_score = scores.min()
scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)     # 归一化
gt_mask = np.asarray(gt_mask_list)
precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
a = 2 * precision * recall
b = precision + recall
f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
threshold = thresholds[np.argmax(f1)]

fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
gt_list = np.asarray(gt_list)
img_roc_auc = roc_auc_score(gt_list, img_scores)
print('image ROCAUC: %.3f' % (img_roc_auc))

plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
plt.legend(loc="lower right")
save_dir = os.path.join(output_path,class_name,'roc_curve_result')
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, class_name + '_roc_curve.png'), dpi=100)
# import pdb;pdb.set_trace()
plot_fig(test_imgs, recon_imgs, scores, gt_mask_list, threshold, save_dir,class_name)