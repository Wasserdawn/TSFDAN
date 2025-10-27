from __future__ import division
import time
import datetime
import argparse
import numpy as np
from loader import get_loader

import torch
from torch.optim import lr_scheduler
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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] ="1"

parser = argparse.ArgumentParser()
parser.add_argument('--LoRA_mode', type=str, default="load", help="none, load")
parser.add_argument('--LoRA_path', type=str, default="lora_weights_minfound/distill/LoRA_6.pkl")
parser.add_argument('--LoRA_save_path', type=str, default=r"D:/ysm/code/code_new/Neighbor2Neighbor-main/boneCT/ckpt")
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
parser.add_argument('--val_dirs', type=str, default='./validation')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
parser.add_argument('--gpu_devices', default='0', type=str)
# parser.add_argument('--parallel', action='store_true')
parser.add_argument('--parallel', type=bool, default=False)
parser.add_argument('--n_feature', type=int, default=48)
parser.add_argument('--n_channel', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=1)
# parser.add_argument('--batchsize', type=int, default=16)
# parser.add_argument('--patchsize', type=int, default=256)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--increase_ratio", type=float, default=2.0)

parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--load_mode', type=int, default=0)
parser.add_argument('--input_path', type=str, default=r'C:\Users\PS\Desktop\ysm\minfoundsyn\test')
parser.add_argument('--target_path', type=str, default=r'C:\Users\PS\Desktop\ysm\minfoundsyn\test')
parser.add_argument('--eval_input_path', type=str, default=r'C:\Users\PS\Desktop\BoneCT\boneCT_sin_test')
parser.add_argument('--eval_target_path', type=str, default=r'C:\Users\PS\Desktop\BoneCT\boneCT_sin_test')
parser.add_argument('--save_path', type=str, default=r'D:\ysm\code\code_new\Neighbor2Neighbor-main\boneCT')
parser.add_argument('--test_patient', type=str, default='TEST')
parser.add_argument('--result_fig', type=bool, default=True)
parser.add_argument('--norm_range_min', type=float, default=-1024.0)
parser.add_argument('--norm_range_max', type=float, default=3072.0)
parser.add_argument('--trunc_min', type=float, default=-1024.0)
parser.add_argument('--trunc_max', type=float, default=3072.0)
parser.add_argument('--transform', type=bool, default=False)
parser.add_argument('--patch_n', type=int, default=8)  # 10
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=6)  # 16
parser.add_argument('--num_workers', type=int, default=0)

opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0

def load_model():
    model = SwinIR()
    state_dict = torch.load(r'SwinTransformer_288800iter.ckpt')
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
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
        target_modules=target_modules_txt,
        # modules_to_save=["stage_2.conv_easy"],
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


def save_fig(x, y, pred, fig_name, original_result, pred_result, trunc_min = -160.0, trunc_max = 240.0):
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

    # f.savefig(os.path.join(r'E:\yuansimiao\data_article\proposed', 'fig', name[0].replace('npy','png')))
    f.savefig(os.path.join(opt.save_path, 'fig','result_{}.png'.format(fig_name)))
    plt.close()



def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))

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

num_epoch = opt.n_epoch


optimizer = Adan(network.parameters(), lr=0.0005, weight_decay=0.02, betas=[0.98, 0.92, 0.99], eps = 1e-8, max_grad_norm=0.0, no_prox=False)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print('init finish')

criterion = nn.L1Loss()


# 训练
if opt.mode == 'train':
    for epoch in range(1, opt.n_epoch + 1):
        cnt = 0

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

        network.train()
        for iteration, (noisy, pseudo) in enumerate(data_loader):
            st = time.time()
            noisy = noisy.float().to(device="cuda")
            noisy = noisy.view(-1, 1, opt.patch_size, opt.patch_size)
            pseudo = pseudo.float().to(device="cuda")
            pseudo = pseudo.view(-1, 1, opt.patch_size, opt.patch_size)


            noisy_denoised = network(noisy)
            optimizer.zero_grad()
            network.zero_grad()

            loss = criterion(noisy_denoised, pseudo)

            loss.backward()
            optimizer.step()
            # if iteration % 20 == 0:
            print(
                '{:04d} {:05d} Loss1={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss.item()),
                        time.time() - st))

        scheduler.step()

        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            LoRA_state_dict = get_peft_model_state_dict(network)
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

                shape_ = x.shape[-1]
                x = trunc(denormalize_(x.view(shape_, shape_).cpu().detach(), opt), opt)
                y = trunc(denormalize_(y.view(shape_, shape_).cpu().detach(), opt), opt)
                prediction = trunc(denormalize_(prediction.view(shape_, shape_).cpu().detach(), opt), opt)
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
            with open(r'lora_weights_timo\eval.txt', 'a') as f:
                f.write('After: {:.4f} '.format(epoch))
                f.write('PSNR avg: {:.4f}   SSIM avg: {:.4f}   RMSE avg: {:.4f}\n'.format(
                    pred_psnr_avg / len(data_loader_val),
                    pred_ssim_avg / len(data_loader_val),
                    pred_rmse_avg / len(data_loader_val),
                ))






# 训练
if opt.mode == 'test':
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

        shape_ = x.shape[-1]
        x = trunc(denormalize_(x.view(shape_, shape_).cpu().detach(),opt),opt)
        y = trunc(denormalize_(y.view(shape_, shape_).cpu().detach(),opt),opt)
        prediction = trunc(denormalize_(prediction.view(shape_, shape_).cpu().detach(),opt),opt)

        np.save(os.path.join(opt.save_path, 'x', '{}_result'.format(i)), x)
        np.save(os.path.join(opt.save_path, 'y', '{}_result'.format(i)), y)
        np.save(os.path.join(opt.save_path, 'pred', '{}_result'.format(i)), prediction)
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
