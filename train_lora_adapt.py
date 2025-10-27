from __future__ import division
import time
import glob
import datetime
import argparse
import numpy as np
from loader import get_loader

import cv2
from PIL import Image
import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

from new_model import SwinIR
from measure import compute_measure

from adan import Adan
from peft import get_peft_model_state_dict
from peft import LoraConfig
from peft import get_peft_model
from peft import set_peft_model_state_dict

from torch import nn
import os
from torchvision.models import vgg19

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] ="1"

parser = argparse.ArgumentParser()
parser.add_argument('--LoRA_mode', type=str, default="load", help="none, load")
parser.add_argument('--LoRA_path', type=str, default=r"D:/ysm/code/code_new/Neighbor2Neighbor-main/boneCT/ckpt/LoRA_100.pkl")
parser.add_argument('--LoRA_save_path', type=str, default=r"D:/ysm/code/code_new/Neighbor2Neighbor-main/boneCT/ckpt")
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
# parser.add_argument('--parallel', action='store_true')
parser.add_argument('--parallel', type=bool, default=True)
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_epoch_100', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=5)
# parser.add_argument('--batchsize', type=int, default=16)
# parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--load_mode', type=int, default=0)
parser.add_argument('--input_path', type=str, default=r'C:\Users\PS\Desktop\BoneCT\boneCT_sin_test')
parser.add_argument('--target_path', type=str, default=r'C:\Users\PS\Desktop\BoneCT\boneCT_sin_test')
parser.add_argument('--eval_input_path', type=str, default=r'C:\Users\PS\Desktop\BoneCT\boneCT_sin_test')
parser.add_argument('--eval_target_path', type=str, default=r'C:\Users\PS\Desktop\BoneCT\boneCT_sin_test')
parser.add_argument('--save_path', type=str, default=r'D:\ysm\code\code_new\Neighbor2Neighbor-main\boneCT')
parser.add_argument('--test_patient', type=str, default='TEST')
parser.add_argument('--result_fig', type=bool, default=True)
parser.add_argument('--norm_range_min', type=float, default=0.0)
parser.add_argument('--norm_range_max', type=float, default=5.0)
parser.add_argument('--trunc_min', type=float, default=0.0)
parser.add_argument('--trunc_max', type=float, default=5.0)
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--patch_n', type=int, default=10)  # 10
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=8)  # 16
parser.add_argument('--num_workers', type=int, default=0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0

def load_model():
    model = SwinIR()
    state_dict = torch.load('SwinTransformer_288800iter.ckpt')
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    with open('config/lora.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    modules_list = content.strip().replace("'", "").split(',\n')
    target_modules_txt = []
    for module in modules_list:
        target_modules_txt.append(module)
    config = LoraConfig(
        r=16,
        lora_alpha=16, # 16
        target_modules=["qkv", "proj", "fc1", "fc2","conv_after_body","conv_last"]
        # target_modules=target_modules_txt,
    )
    model = get_peft_model(model, config)

    # model.to(f"cuda:{opt.cuda_index}")
    model.to(device='cuda')
    return model


def load_model_test(opt):
    state_dict = torch.load("SwinTransformer_288800iter.ckpt")
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model = SwinIR()
    model.load_state_dict(state_dict)
    # print(model)

    if opt.LoRA_mode == "load":
        with open('config/lora.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        modules_list = content.strip().replace("'", "").split(',\n')
        target_modules_txt = []
        for module in modules_list:
            target_modules_txt.append(module)
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=target_modules_txt,
            # modules_to_save=["stage_2.conv_easy"],
        )
        model = get_peft_model(model, config)

        lora_state_dict = torch.load(opt.LoRA_path, map_location="cpu")
        set_peft_model_state_dict(model, lora_state_dict)

    model.to(device='cuda')
    model.eval()
    return model


def reverse_pixelshuffle(image, scale, fill=0, fill_image=0, ind=[0, 0]):
    '''
    Discription: Given a mosaic image of subsampling, recombine it to a full image
    [Input]: Image
    [Return]: Recombine it using different portions of pixels
    '''
    w, h = image.shape
    real = np.zeros((w, h))  # real image
    wf = 0
    hf = 0
    for ws in range(scale):
        hf = 0
        for hs in range(scale):
            temp = real[ws::scale, hs::scale]
            wc, hc = temp.shape  # get the shpae of the current images
            if fill == 1 and ws == ind[0] and hs == ind[1]:
                real[ws::scale, hs::scale] = fill_image[wf:wf + wc, hf:hf + hc]
            else:
                real[ws::scale, hs::scale] = image[wf:wf + wc, hf:hf + hc]
            hf = hf + hc
        wf = wf + wc
    return real


def save_fig(x, y, pred, fig_name, original_result, pred_result, trunc_min = -160, trunc_max = 240.0):
    x, y, pred = x.numpy(), y.numpy(), pred.numpy()
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                       original_result[1],
                                                                       original_result[2],
                                                                       ), fontsize=20)#\nLPIPS:{:.4f},original_result[3]
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                       pred_result[1],
                                                                       pred_result[2],
                                                                       ), fontsize=20)#\nLPIPS:{:.4f},
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('Full-dose', fontsize=30)

    f.savefig(os.path.join(opt.save_path, 'fig', 'result_{}.png'.format(fig_name)))
    plt.close()



def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_dir, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_dir = data_dir
        self.patch = patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        # np.ndarray to torch.tensor
        transformer = transforms.Compose([transforms.ToTensor()])
        im = transformer(im)
        return im

    def __len__(self):
        return len(self.train_fns)


def validation_kodak(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_bsd300(dataset_dir):
    fns = []
    fns.extend(glob.glob(os.path.join(dataset_dir, "test", "*")))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def validation_Set14(dataset_dir):
    fns = glob.glob(os.path.join(dataset_dir, "*"))
    fns.sort()
    images = []
    for fn in fns:
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        images.append(im)
    return images


def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


def denormalize_(image,opt):
    image = image * (opt.norm_range_max - opt.norm_range_min) + opt.norm_range_min
    return image


def trunc(mat,opt):
    mat[mat <=opt.trunc_min] = opt.trunc_min
    mat[mat >= opt.trunc_max] = opt.trunc_max
    return mat


# Training Set
data_loader = get_loader(mode=opt.mode,
                         load_mode=opt.load_mode,
                         # saved_path=opt.saved_path,
                         input_path=opt.input_path,
                         target_path=opt.target_path,
                         test_patient=opt.test_patient,
                         patch_n=(opt.patch_n if opt.mode=='train' else None),
                         patch_size=(opt.patch_size if opt.mode=='train' else None),
                         transform=opt.transform,
                         batch_size=(opt.batch_size if opt.mode=='train' else 1),
                         num_workers=opt.num_workers)
data_loader_val = get_loader(mode="test",
                         load_mode=opt.load_mode,
                         # saved_path=opt.saved_path_val,
                         input_path=opt.eval_input_path,
                         target_path=opt.eval_target_path,
                         test_patient=opt.test_patient,
                         patch_n=None,
                         patch_size=None,
                         transform=opt.transform,
                         batch_size= 1,
                         num_workers=opt.num_workers)


if opt.mode == 'train':
    network = load_model()
if opt.mode == 'test':
    network = load_model_test(opt)

if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
param = get_parameter_number(network)
print('param=', param)

# about training scheme
num_epoch = opt.n_epoch


optimizer = Adan(network.parameters(), lr=0.0005, weight_decay=0.02, betas=[0.98, 0.92, 0.99], eps = 1e-8, max_grad_norm=0.0, no_prox=False)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print('init finish')



# 训练
if opt.mode == 'train':
    for epoch in range(1, opt.n_epoch + 1):
        cnt = 0

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

        network.train()
        for iteration, noisy in enumerate(data_loader):
            st = time.time()
            noisy = noisy.float().to(device="cuda")
            noisy = noisy.view(-1, 1, opt.patch_size, opt.patch_size)

            optimizer.zero_grad()

            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)

            with torch.no_grad():
                noisy_denoised = network(noisy)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2
            Lambda = epoch / opt.n_epoch_100 * opt.increase_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2

            loss_all.backward()
            optimizer.step()
            print(
                '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                        np.mean(loss2.item()), np.mean(loss_all.item()),
                        time.time() - st))

        scheduler.step()

        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            LoRA_state_dict = get_peft_model_state_dict(network.module)
            LoRA_state_path = os.path.join(opt.LoRA_save_path, f"LoRA_{epoch}.pkl")
            torch.save(LoRA_state_dict, LoRA_state_path)

            ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
            pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
            ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
            pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []

            for i, (x, y) in enumerate(data_loader_val):
                print(i)
                psnr_result = []
                ssim_result = []
                x = x.unsqueeze(0).float().to(device="cuda")
                y = y.unsqueeze(0).float().to(device="cuda")
                with torch.no_grad():
                    prediction = network(x)
                shape0, shape1 = x.shape[-2], x.shape[-1]
                x = trunc(denormalize_(x.view(shape0, shape1).cpu().detach(), opt), opt)
                y = trunc(denormalize_(y.view(shape0, shape1).cpu().detach(), opt), opt)
                prediction = trunc(denormalize_(prediction.view(shape0, shape1).cpu().detach(), opt), opt)

                x = reverse_pixelshuffle(x, 2)
                y = reverse_pixelshuffle(y, 2)
                prediction = reverse_pixelshuffle(prediction, 2)

                x = torch.from_numpy(x)
                y = torch.from_numpy(y)
                prediction = torch.from_numpy(prediction)
                data_range = opt.trunc_max - opt.trunc_min

                original_result, pred_result = compute_measure(x, y, prediction, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(data_loader_val))
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(data_loader_val))
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(data_loader_val))
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(data_loader_val))
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(data_loader_val))
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(data_loader_val))

            print('\n')
            print(
                'Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                    ori_psnr_avg / len(data_loader_val),
                    ori_ssim_avg / len(data_loader_val),
                    ori_rmse_avg / len(data_loader_val),
                ))
            print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(data_loader_val),
                pred_ssim_avg / len(data_loader_val),
                pred_rmse_avg / len(data_loader_val),
            ))
            with open(r'D:\ysm\code\code_new\Neighbor2Neighbor-main\boneCT\eval.txt', 'a') as f:
                f.write('After: {:.4f} '.format(epoch))
                f.write('PSNR avg: {:.4f}   SSIM avg: {:.4f}   RMSE avg: {:.4f}\n'.format(
                    pred_psnr_avg / len(data_loader_val),
                    pred_ssim_avg / len(data_loader_val),
                    pred_rmse_avg / len(data_loader_val),
                ))






# 训练
if opt.mode == 'test':
    # for valid_name, valid_images in valid_dict.items():
    # network.load_state_dict(torch.load(r"D:\ysm\code\Neighbor2Neighbor-main\Neighbor2Neighbor-main\results\unet_gauss25_b4e100r02\2024-12-26-19-04\epoch_model_100.pth"))

    ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
    ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
    pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []

    for i, (x, y) in enumerate(data_loader_val):
        print(i)
        psnr_result = []
        ssim_result = []
        x = x.unsqueeze(0).float().to(device="cuda")
        y = y.unsqueeze(0).float().to(device="cuda")
        with torch.no_grad():
            prediction = network(x)
        shape0, shape1 = x.shape[-2], x.shape[-1]
        x = trunc(denormalize_(x.view(shape0, shape1).cpu().detach(),opt),opt)
        y = trunc(denormalize_(y.view(shape0, shape1).cpu().detach(),opt),opt)
        prediction = trunc(denormalize_(prediction.view(shape0, shape1).cpu().detach(),opt),opt)

        x = reverse_pixelshuffle(x, 2)
        y = reverse_pixelshuffle(y, 2)
        prediction = reverse_pixelshuffle(prediction, 2)



        # np.save(os.path.join(opt.save_path, 'x', '{}_result'.format(i)), x)
        # np.save(os.path.join(opt.save_path, 'y', '{}_result'.format(i)), y)
        np.save(os.path.join(opt.save_path, 'pred', '{}_result'.format(i)), prediction)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        prediction = torch.from_numpy(prediction)
        data_range = opt.trunc_max - opt.trunc_min

        original_result, pred_result = compute_measure(x, y, prediction, data_range)
        ori_psnr_avg += original_result[0]
        ori_psnr_avg1.append(ori_psnr_avg / len(data_loader_val))
        ori_ssim_avg += original_result[1]
        ori_ssim_avg1.append(ori_ssim_avg / len(data_loader_val))
        ori_rmse_avg += original_result[2]
        ori_rmse_avg1.append(ori_rmse_avg / len(data_loader_val))
        pred_psnr_avg += pred_result[0]
        pred_psnr_avg1.append(pred_psnr_avg / len(data_loader_val))
        pred_ssim_avg += pred_result[1]
        pred_ssim_avg1.append(pred_ssim_avg / len(data_loader_val))
        pred_rmse_avg += pred_result[2]
        pred_rmse_avg1.append(pred_rmse_avg / len(data_loader_val))

        if opt.result_fig:
            save_fig(x, y, prediction, i, original_result, pred_result)

    print('\n')
    print(
        'Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg / len(data_loader_val),
                                                                                  ori_ssim_avg / len(data_loader_val),
                                                                                  ori_rmse_avg / len(data_loader_val),
                                                                                  ))
    print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
        pred_psnr_avg / len(data_loader_val),
        pred_ssim_avg / len(data_loader_val),
        pred_rmse_avg / len(data_loader_val),
        ))
