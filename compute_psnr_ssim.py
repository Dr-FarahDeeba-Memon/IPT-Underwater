import os
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

compute_finetuned = True

def compute_avg_psnr_ssim(true_imgs_path, test_imgs_path):
    true_image_paths = sorted(glob.glob(f"{true_imgs_path}/*"))
    test_image_paths = sorted(glob.glob(f"{test_imgs_path}/*"))

    assert len(true_image_paths) == len(test_image_paths)

    n_samples = len(true_image_paths)

    sum_psnr, sum_ssim = 0, 0

    for i in range(n_samples):
        assert true_image_paths[i].split("/")[-1] == test_image_paths[i].split("/")[-1]

        # Read true and test images
        true_img = cv2.imread(true_image_paths[i])
        test_img = cv2.imread(test_image_paths[i])

        assert true_img.shape == test_img.shape

        # Compute the psnr and ssim and add to the total
        sum_psnr += calc_psnr(true_img, test_img)
        sum_ssim += calc_ssim(true_img, test_img, channel_axis=2)

        if (i+1)%200 == 0:
            print(f"....{i+1}/{n_samples} computed.")

    # Compute and return the average psnr and ssim for the whole dataset
    avg_psnr, avg_ssim = sum_psnr/n_samples, sum_ssim/n_samples
    return np.around(avg_psnr, decimals=2), np.around(avg_ssim, decimals=4)

if __name__ == '__main__':
    dataset_path = 'EUVP/test_25'
    true_imgs_path = f"benchmark/{dataset_path}/GTr"

    test_imgs_path = f"benchmark/{dataset_path}/Inp"
    avg_psnr, avg_ssim = compute_avg_psnr_ssim(true_imgs_path, test_imgs_path)
    print("Before IPT (Noisy): EUVP Test Dataset")
    print(f"Noisy Average PSNR: {avg_psnr}")
    print(f"Noisy Average SSIM: {avg_ssim}")


    test_imgs_path = f"./results/{dataset_path}"
    avg_psnr, avg_ssim = compute_avg_psnr_ssim(true_imgs_path, test_imgs_path)
    print("After Pretrained IPT: EUVP Test Dataset")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")

    if compute_finetuned:
        # Read all finetuned models
        all_models = sorted(glob.glob("finetuned_models/*.pth"))
        #n_models = len(all_models)
        n_models = sorted([int(m.split("_")[-1].split(".")[0]\
                        .split("e")[-1]) for m in all_models])[-1]

        # Compute PSNR and SSIM of finetuned models
        plot_eps, avg_psnrs, avg_ssims = [], [], []
        for i in range(n_models):
            if (i+1)==1 or (i+1)%1==0:
                m = f"finetuned_models/model_e{i+1}.pth"
                if os.path.exists(m):
                    print(f"Computing result of finetuned model {i+1} of {n_models}")
                    model_name = (m.split(".")[0]).split("/")[-1]

                    test_imgs_path = f"./results/{dataset_path}_finetuned/{model_name}"
                    avg_psnr, avg_ssim = compute_avg_psnr_ssim(true_imgs_path, test_imgs_path)
                    avg_psnrs.append(avg_psnr)
                    avg_ssims.append(avg_ssim)
                    plot_eps.append(i+1)
                    print(f"Finetuned IPT model {model_name}: EUVP Test Dataset")
                    print(f"Average PSNR: {avg_psnr}")
                    print(f"Average SSIM: {avg_ssim}")
        
        plt.figure()
        plt.plot(np.asarray(plot_eps, dtype='int'),
                np.asarray(avg_psnrs, dtype='float32'))
        plt.title("Test PSNR (EUVP)")
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.savefig("test_PSNR.png")

        plt.figure()
        plt.plot(np.asarray(plot_eps, dtype='int'),
                np.asarray(avg_ssims, dtype='float32'))
        plt.title("Test SSIM (EUVP)")
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.savefig("test_SSIM.png")
        




