import os
import torch
import cv2

import numpy as np
import glob
import os
from multiprocessing import Pool
from itertools import repeat

import torch.utils.data as data
import glob

patch_size = 48

def set_requires_grad(model: torch.nn.Module, requires_grad=True):
    def get_all_layers(block):
        # get children form model!
        children = list(block.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return block
        else:
            # look for children from children... to the last child!
            for child in children:
                try:
                    flatt_children.extend(get_all_layers(child))
                except TypeError:
                    flatt_children.append(get_all_layers(child))
        
        return flatt_children

    total_params = 0
    for l in get_all_layers(model):        # Set requires_grad
        for param in l.parameters():
            param.requires_grad = requires_grad
            total_params += param.numel()
    
    return total_params

def extract_patches(img_path, save_dir, patch_size):
    img = cv2.imread(img_path)

    cx, cy = int(img.shape[0]/2), int(img.shape[1]/2)
    img = img[cx-48:cx+48, cy-48:cy+48, :]

    img_name = (img_path.split('/')[-1]).split('.')[0]

    for i in range(2):
        for j in range(2):
            patch_path = save_dir + "/" + img_name + f"_{i}_{j}" + ".jpg"

            if not os.path.exists(patch_path):
                patch = img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
                cv2.imwrite(patch_path, patch)


def generate_data(train_sets :list):

    # Extract the patches
    for set in train_sets:
        dataset_dir = f'EUVP/Paired/{set}'
        noisy_euvp_dir = f"benchmark/{dataset_dir}/trainA"     # Path to EUVP train data
        clean_euvp_dir = f"benchmark/{dataset_dir}/trainB"

        noisy_train_dir = f"train_data/EUVP/{set}/input"
        clean_train_dir = f"train_data/EUVP/{set}/output"

        # Create training data directories if it does not exist
        if not os.path.exists(noisy_train_dir):
            os.makedirs(noisy_train_dir)

        if not os.path.exists(clean_train_dir):
            os.makedirs(clean_train_dir)

        # read all image paths
        noisy_image_paths = sorted(glob.glob(f"{noisy_euvp_dir}/*"))
        clean_image_paths = sorted(glob.glob(f"{clean_euvp_dir}/*"))

        # Generate noisy patches
        with Pool(processes=1) as pool:
            pool.starmap(extract_patches,  
                    zip(noisy_image_paths, repeat(noisy_train_dir), repeat(patch_size)))
        
        # Generate clean patches
        with Pool(processes=1) as pool:
            pool.starmap(extract_patches,  
                    zip(clean_image_paths, repeat(clean_train_dir), repeat(patch_size)))
        print(f"..... {set} completed.")


class EuVPDataset(data.Dataset):
    def __init__(self, base_folder_path, input_folder_name, output_folder_name, img_size=None):
        super(EuVPDataset, self).__init__()
        self.noisy_files = glob.glob(os.path.join(base_folder_path,
                                        input_folder_name,'*.jpg'))
        self.clean_files = []
        for img_path in self.noisy_files:
             self.clean_files.append(os.path.join(base_folder_path,
                                    output_folder_name, os.path.basename(img_path))) 
        
        self.img_size = img_size

    def __getitem__(self, index):
            noisy_path = self.noisy_files[index]
            clean_path = self.clean_files[index]
            n_img = cv2.imread(noisy_path)
            c_img = cv2.imread(clean_path)

            if self.img_size is not None:
                n_img = cv2.resize(n_img, (self.img_size, self.img_size))
                c_img = cv2.resize(c_img, (self.img_size, self.img_size))
            noisy_img = np.moveaxis(n_img, -1, 0)
            clean_img = np.moveaxis(c_img, -1, 0)
            return torch.from_numpy(noisy_img).float(), torch.from_numpy(clean_img).float()

    def __len__(self):
        return len(self.noisy_files)
    