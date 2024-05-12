import torch
import os
import matplotlib.pyplot as plt
import time
from utils.data_loader import DataLoader
from utils.save_results import save_images, save_video
import numpy as np
from utils.feature_field import FeatureField, normalize_coordinates
torch.cuda.empty_cache()
os.chdir("models/slang_mlp")
from models.slang_mlp.image_model import RenderImage
os.chdir("../..")

class SlangHashTrainer:
    def __init__(self):
        self.width, self.height = 128, 128
        self.N_samples = 32
        self.C = 32
        self.embeded = torch.tensor([True], dtype=torch.bool).cuda()
        self.device = 'cuda'
        self.dataset = DataLoader(H= self.height, W = self.width, N_samples = self.N_samples)
        self.bounding_box = self.dataset.get_bbx()
        self.model = RenderImage()
        self.loss_fn = torch.nn.MSELoss()
        self.feature_field = FeatureField(features_per_level = 2,res=self.height).cuda()
        self.params = self.init_params()
        
    def init_params(self):
        w1 = torch.randn((self.C, self.C), dtype=torch.float, requires_grad=True, device='cuda:0')
        w2 = torch.randn((self.C, self.C), dtype=torch.float, requires_grad=True, device='cuda:0')
        w3 = torch.randn((self.C, self.C), dtype=torch.float, requires_grad=True, device='cuda:0')
        b1 = torch.zeros(self.C, dtype=torch.float, requires_grad=True, device='cuda:0')
        b2 = torch.zeros(self.C, dtype=torch.float, requires_grad=True, device='cuda:0')
        b3 = torch.zeros(self.C, dtype=torch.float, requires_grad=True, device='cuda:0')
        return [w1, w2, w3, b1, b2, b3]
    
    def train(self, iters, lr=5e-3):
        self.iters = iters
        #optimizer = torch.optim.Adam(self.params, lr=lr) 
        optimizer = torch.optim.Adam([ {'params': self.params},
                                {'params': self.feature_field.parameters()}], lr=lr)
        start = time.time()
        for i in range(iters):
            img_i = np.random.randint(100)
            x, dists, target_image, viewdirs = self.dataset.get_data(img_i)
            x = normalize_coordinates(x, self.bounding_box)
            #embedded_x = self.feature_field.encode(x)
            embedded_x = self.feature_field(x)
            
            encoded_viewdirs = viewdirs  
            y_pred = self.model.apply(
            self.width, self.height,
            embedded_x, encoded_viewdirs, dists, self.embeded,
            *self.params)
            
            loss = self.loss_fn(y_pred, target_image)
            psnr = -10. * torch.log(loss) / torch.math.log(10.)
            print(f"Iteration {i}, Loss: {loss.item()}, psnr: {psnr.item()}")            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end = time.time()
        print('avg training time:', (end - start)/iters)
        
    def render(self,saveimg):
        intermediate_images = []
        target_images = []
        psnrs = []
        start = time.time()
        test_images = [100,101,102,103,104,105] #test cases
        for img_i in test_images:
            x, dists, target_image, viewdirs = self.dataset.get_data(img_i)
            x = normalize_coordinates(x, self.bounding_box)
            #embedded_x = self.feature_field.encode(x)   
            embedded_x = self.feature_field(x)

            encoded_viewdirs = viewdirs  
            
            y_pred = self.model.apply(
                self.width, self.height,
                embedded_x, encoded_viewdirs, dists, self.embeded,
                *self.params)
            
            loss = self.loss_fn(y_pred, target_image)
            psnr = -10. * torch.log(loss) / torch.math.log(10.)
            print(f"Iteration {img_i}, Loss: {loss.item()}, psnr: {psnr.item()}") 
            if(saveimg):      
                intermediate_images.append(y_pred.detach().cpu().numpy())
                target_images.append(target_image.detach().cpu().numpy())
                psnrs.append(psnr.detach().cpu())
        end = time.time()
        print('avg rendering time:', (end - start)/len(test_images))
        if(saveimg):
            save_images(target_images, intermediate_images,'slanghash.png', "Slang MLP with hashencoding" , self.iters, psnrs)
        
    def render_path(self, saveimg):
        intermediate_images = []
        for img_i in range(100):
            x, dists, viewdirs = self.dataset.get_render_data(img_i)
            x = normalize_coordinates(x, self.bounding_box)
            #embedded_x = self.feature_field.encode(x)   
            embedded_x = self.feature_field(x)
            encoded_viewdirs = viewdirs  
            y_pred = self.model.apply(
                self.width, self.height,
                embedded_x, encoded_viewdirs, dists, self.embeded,
                *self.params)
            intermediate_images.append(y_pred.detach().cpu().numpy())
            print(f"Iteration {img_i}") 
        save_video(intermediate_images,'slanghash')