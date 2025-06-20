import os 
import sys
sys.path.insert(1, "/".join(os.getcwd().split("/")[:-1]))
import torch
import numpy as np
from dataset import Dataset
from models.model import Model
import torch.nn as nn
import time
import cv2
from torch.utils.data import DataLoader
from train_utils import *
from tqdm import tqdm
from training.data_augmentation import transform_normalize, transform_augment

class Trainer:
    def __init__(self):
        self.lr_min = 1.0e-6
        self.lr_max = 5.0e-4 # base lr for 1 gpu and batch_size=20, before apllied scaling rule
        self.grad_clip_norm = 1.0
        self.n_iters = int(1e4)
        self.n_iters_warmup = int(self.n_iters * 0.1)
        self.device = 7
        self.n_workers = 8

        self.bce_loss_fcn = LossConfidence(reduction="mean")
        self.contrast_loss_fcn = ContrastiveLoss(reduction="mean")
        
        self.scheduler = Scheduler(max_value=self.lr_max, min_value=self.lr_min, num_iters=self.n_iters, num_warmup_iters=self.n_iters_warmup, warmup_start_value=self.lr_min)
        self.grad_scaler = torch.amp.GradScaler(enabled=True)
        
        self.train_statistics = TrainStatistics(loss_reduction_type="mean")

        self.debug = False
        self.model_name = 1

        self.transform_normalize = transform_normalize.to(self.device)
        self.transform_augment = transform_augment.to(self.device)
        
        
        
    def init_model(self):
        model = Model().to(self.device)#.to(torch.float16)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr_min, weight_decay=0.05)

        return model, optimizer


    def one_batch_iteration(self, batch_data, model, optimizer):
        imgs, tars = batch_data
        
        imgs = imgs.to(self.device).squeeze(0)#.to(torch.float16)
        imgs = self.transform_augment(imgs)

        if self.debug:
            imgs = imgs.cpu().numpy() * 255.0
            imgs = imgs.astype(np.uint8)
            os.makedirs("./debug", exist_ok=True)
            for i in range(imgs.shape[0]):
                img = imgs[i].transpose(1, 2, 0)
                cv2.imwrite(f"./debug/{str(i).zfill(3)}.jpg", img)
            exit(0)
            # cv2.destroyAllWindows()
        imgs = self.transform_normalize(imgs)
        
        tars = tars.to(self.device).squeeze(0)#.to(torch.float16)
            
        preds = model(imgs)

        loss_conf, matches_conf = self.bce_loss_fcn(preds, tars)

        loss = loss_conf# + loss_contrast

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm, norm_type=2)
        optimizer.step()

        self.train_statistics.update_stats("conf", loss_conf, matches_conf)
        # self.train_statistics.update_stats("conf_contrast", loss_contrast, matches_contrast)
            
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        

    def init_data_loader(self):
        dataset = Dataset(self.n_iters)

        dataloader = DataLoader(dataset, 
                                batch_size=1, 
                                shuffle=False, 
                                pin_memory=True,
                                num_workers=self.n_workers,
                                persistent_workers=True,
                                )

        return dataloader

    def set_optimizer_params(self, optimizer, iteration):
        lr = self.scheduler.get_value(iteration)
        optimizer.param_groups[0]["lr"] = lr

    def main(self):
        model, optimizer = self.init_model()
        dataloader = self.init_data_loader()
        dataloader = tqdm(dataloader, dynamic_ncols=True)
        for iteration, batch_data in enumerate(dataloader, start=1):
            
            self.set_optimizer_params(optimizer, iteration)
            self.one_batch_iteration(batch_data, model, optimizer)

            if iteration % 100 == 0:
                train_stats = self.train_statistics.get_stats()
                
                print_data = f"B: {iteration}, "
                for key in train_stats.keys():
                    for stat_key in train_stats[key].keys():
                        print_data += f"{key}_{stat_key}: {train_stats[key][stat_key]}, "
                print(print_data)
                self.train_statistics.reset_stats()
                
            if iteration % 100 == 0:
                path2save = f"./weights/{str(self.model_name).zfill(4)}/"
                os.makedirs(path2save, exist_ok=True)
                
                torch.save(model.state_dict(), f"{path2save}/{str(iteration).zfill(6)}.pt")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()
