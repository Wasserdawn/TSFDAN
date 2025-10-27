import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

def pixelshuffle(image, scale):
    '''
    Discription: Given an image, return a reversible sub-sampling
    [Input]: Image ndarray float
    [Return]: A mosic image of shuffled pixels
    '''
    if scale == 1:
        return image
    # w, h, c = image.shape
    mosaic = np.array([])
    for ws in range(scale):
        band = np.array([])
        for hs in range(scale):
            temp = image[ws::scale, hs::scale]  # get the sub-sampled image
            band = np.concatenate((band, temp), axis=1) if band.size else temp
        mosaic = np.concatenate((mosaic, band), axis=0) if mosaic.size else band
    return mosaic

class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, input_path, target_path, test_patient, patch_n=None, patch_size=None, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(input_path, '*_input.nii')))
        if mode=='train':
            target_path = sorted(glob(os.path.join(target_path, '*_input.nii')))
        if mode == 'test':
            target_path = sorted(glob(os.path.join(target_path, '*_pred.nii.gz')))
        # saved_path = r'C:\Users\PS\Desktop\ysm\val\val'
        # input_path = sorted(glob(os.path.join(input_path, '*_input.nii')))
        # target_path = sorted(glob(os.path.join(target_path, '*_pred.nii.gz')))
        # input_path = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        # target_path = [os.path.join(target_path, f) for f in os.listdir(target_path)]
        self.input_ = input_path
        self.target_ = target_path
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        file_name = os.path.basename(input_img)
        # if self.load_mode == 0:
        # input_img, target_img = np.load(input_img), np.load(target_img)
        input_img, target_img = sitk.ReadImage(input_img), sitk.ReadImage(target_img)
        input_img, target_img = sitk.GetArrayFromImage(input_img), sitk.GetArrayFromImage(target_img)
        input_img, target_img = normalize_(input_img), normalize_(target_img)
        # input_img, target_img = input_img, target_img
        input_img, target_img = pixelshuffle(input_img, 2), pixelshuffle(target_img, 2)


        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches)
        else:
            # return (input_img, target_img, file_name)
            return (input_img, target_img)


def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(mode='train', load_mode=0,
               input_path = None, target_path = None, test_patient='LDCT',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, num_workers=6):
    dataset_ = ct_dataset(mode, load_mode, input_path, target_path, test_patient, patch_n, patch_size, transform)
    #喂ndarray、list、或者tuple等给dataset，dataloader会将其自动转换为tensor
    #dataset_中是array
    # for i in dataset_:
        # print(i)
    data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False, num_workers=num_workers)   # shuffle=True
    #data_loader中是tensor
    # for i in data_loader:
        # print(i)
        # print(type(i))
        # print(i[0].shape)
    return data_loader

def normalize_(image, MIN_B=0.0, MAX_B=5.0):#-1024,3072 # 1500，6000
   image = (image - MIN_B) / (MAX_B - MIN_B)
   return image
