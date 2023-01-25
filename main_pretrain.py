# IPT main file for frames enhancement of a set of images

from option import args

import torch
import glob
import utility_pretrain
import data
import loss
import warnings
warnings.filterwarnings('ignore')
import os
os.system('pip install einops')
import model
torch.manual_seed(args.seed)

from trainer_pretrain import Trainer
checkpoint = utility_pretrain.checkpoint(args)

checkpoint.args.test_only = True
checkpoint.args.dir_data = "CBSD68"    # This does not matter
checkpoint.args.pretrain = "pretrained_models/IPT_denoise30.pt"
#checkpoint.args.pretrain = "pretrained_models/IPT_pretrain.pt"
checkpoint.args.data_test = ["CBSD68"]    # This does not matter as well
checkpoint.args.scale = [1]
checkpoint.args.denoise = True
checkpoint.args.cpu = False
checkpoint.args.n_GPUs = 1

if 'IPT_pretrain' in args.pretrain:
    args.num_queries = 6

dataset_path = 'EUVP/test_25'
IMG_PATH = f'./benchmark/{dataset_path}/Inp/*'      # Path to our noisy EUVP test dataset
img_names = sorted(glob.glob(IMG_PATH))

save_dir = f"./results/{dataset_path}"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def main():
    global model
    if checkpoint.ok:
        loader = data.Data(args)
        
        loader.loader_test[0].dataset.images_hr = img_names
        loader.loader_test[0].dataset.images_lr = img_names
        
        _model = model.Model(args, checkpoint)
        state_dict = torch.load(args.pretrain)
        _model.model.load_state_dict(state_dict,strict = False)
        #_model = torch.load(args.pretrain)
        _model.n_GPUs = checkpoint.args.n_GPUs
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        t.test(save_dir)
        checkpoint.done()
        
        print('Done')
            
if __name__ == '__main__':
    main()   
