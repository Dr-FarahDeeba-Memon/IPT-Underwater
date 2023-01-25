# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import utility
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb
import os
import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def test(self, save_dir, gt_rects=None, exp_ratio=0.5):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                
                i = 0
                for hr, _,filename in tqdm(d, ncols=80):
                    vid_name = (save_dir.split('/')[-2]).split('_')[-1]
                    print("--", filename[0], "/", vid_name)
                    
                    name = str(filename[0].split('/')[-1].split('_x')[0])+ '.jpg'
                    save_path = save_dir + '/' + name
                    
                    if not os.path.exists(save_path):
                        if gt_rects is not None:
                            gt_rect = gt_rects[i]
                            
                            # Initialize sr to the original image
                            sr = hr
                            
                            hr_np = hr[0].cpu().numpy()
                            tot_w, tot_h = hr_np.shape[2], hr_np.shape[1]
                            
                            # Crop the image around the target
                            x,y,w,h = gt_rect
                            
                            x_start, y_start = int(x-(exp_ratio*w)), int(y-(exp_ratio*h))
                            x_stop, y_stop = int(x+w+(exp_ratio*w)), int(y+h+(exp_ratio*h)) 
                            
                            min_win = 100
                            if x_stop-x_start < min_win: # Keep minimum dimension 50 kernel size is 48
                                diff = min_win - (x_stop-x_start)
                                diff_2 = int(diff/2)
                                x_start, x_stop = x_start-diff_2, x_stop+diff_2
                            
                            if y_stop-y_start < min_win: # Keep minimum dimension 50 kernel size is 48
                                diff = min_win - (y_stop-y_start)
                                diff_2 = int(diff/2)
                                y_start, y_stop = y_start-diff_2, y_stop+diff_2
                            
                            x_start = 0 if x_start < 0 else x_start
                            y_start = 0 if y_start < 0 else y_start
                            x_stop = x_stop if x_stop < tot_w else tot_w
                            y_stop = y_stop if y_stop < tot_h else tot_h
                            
                            
                            hr_crop = hr_np[:, y_start:y_stop, x_start:x_stop]
                            hr_crop = np.expand_dims(hr_crop, axis=0)
                            hr_crop_tensor = torch.from_numpy(hr_crop)
                            
                            # Perform enhancement on croped part
                            hr_crop = self.prepare(hr_crop_tensor)[0]
                            sr_crop = self.model(hr_crop, idx_scale)
                            
                            # Replace in original hr
                            sr[:, :, y_start:y_stop, x_start:x_stop] = sr_crop
                        else:
                            hr = self.prepare(hr)[0]
                            sr = self.model(hr, idx_scale)
                        
                        # Quantize and save result
                        sr = utility.quantize(sr, self.args.rgb_range)
                        
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                                    sr, hr, scale, self.args.rgb_range)
                        self.ckp.save_results(d, filename[0], sr, 50, save_path)
                    i = i+1
                
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs