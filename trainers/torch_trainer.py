import torch
import os
import time
from utils.rendering_utils import volume_rendering
from utils.data_loader import DataLoader
from utils.encoder import pos_embed
from utils.save_results import save_images, save_video
import numpy as np
from models.torch_mlp.mlp import MLP
torch.cuda.empty_cache()

class TorchTrainer:
    def __init__(self):
        self.width, self.height = 128, 128
        self.N_samples = 32
        self.C = 32
        self.device = 'cuda'
        self.model = MLP(self.C, self.C, self.C).to(self.device)
        self.dataset = DataLoader(H= self.height, W = self.width, N_samples = self.N_samples)
        self.loss_fn = torch.nn.MSELoss()
        
    def train(self, iters, lr = 5e-3):
        self.iters = iters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        start = time.time()
        for i in range(iters):
            img_i = np.random.randint(100)
            x, dists, target_image, viewdirs = self.dataset.get_data(img_i)
            x = pos_embed(x)[...,:-1]
            viewdirs = (pos_embed(viewdirs)[...,:16]).unsqueeze(2).expand(-1, -1, self.N_samples, -1)
            y_pred = self.model(x, viewdirs)
            y_pred = volume_rendering(y_pred, dists)
            loss = self.loss_fn(y_pred, target_image)
            psnr = -10. * torch.log(loss) / torch.math.log(10.)
            print(f"Iteration {i}, Loss: {loss.item()}, psnr: {psnr.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end = time.time()
        print('avg training time:', (end - start)/iters)
        
    def render(self, saveimg):
        intermediate_images = []
        target_images = []
        psnrs = []
        test_images = [100,101,102,103,104,105]
        for img_i in test_images:
            x, dists, target_image, viewdirs = self.dataset.get_data(img_i)
            x = pos_embed(x)[...,:-1]
            viewdirs = (pos_embed(viewdirs)[...,:16]).unsqueeze(2).expand(-1, -1, self.N_samples, -1)
            y_pred = self.model(x, viewdirs)
            y_pred = volume_rendering(y_pred, dists)
            loss = self.loss_fn(y_pred, target_image)
            psnr = -10. * torch.log(loss) / torch.math.log(10.)
            print(f"Iteration {img_i}, Loss: {loss.item()}, psnr: {psnr.item()}")
            if(saveimg):
                intermediate_images.append(y_pred.detach().cpu().numpy())
                target_images.append(target_image.detach().cpu().numpy())
                psnrs.append(psnr.detach().cpu())
        if(saveimg):
            save_images(target_images, intermediate_images,'torch.png', "PyTorch MLP" , self.iters,psnrs)
            
    def render_path(self, saveimg):
        intermediate_images = []
        start = time.time()
        for img_i in range(100):
            x, dists, viewdirs = self.dataset.get_render_data(img_i)
            x = pos_embed(x)[...,:-1]
            viewdirs = (pos_embed(viewdirs)[...,:16]).unsqueeze(2).expand(-1, -1, self.N_samples, -1)
            y_pred = self.model(x, viewdirs)
            y_pred = volume_rendering(y_pred, dists)     
            intermediate_images.append(y_pred.detach().cpu().numpy())
            print(f"Iteration {img_i}") 
        end = time.time()
        print('avg rendering time:', (end - start)/100)
        save_video(intermediate_images,'torch')