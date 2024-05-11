import numpy as np
import torch
import cv2
import time

class DataLoader():
    def __init__(self, H, W, N_samples = 32):
        self.device = "cuda"
        self.H, self.W = H, W
        self.near= 2
        self.far = 6
        self.N_samples = N_samples
        self.load_data()
        
    def load_data(self):   
        data = np.load('data/dataset.npz')
        images = np.array(data['images'], dtype= np.float32)[...,:3]
        poses = torch.tensor(data['poses'], dtype=torch.float32, device=self.device)
        focal = torch.tensor(data['focal'], dtype=torch.float32, device=self.device).clone().detach()
        self.preprocess_data(images, poses, focal)
        self.bbx = self.get_bbx()
        
    def preprocess_data(self, images, poses, focal):
        num_images = images.shape[0]
        self.target_images = np.zeros((num_images, self.H, self.W, 3), dtype= np.float32)
        self.distances = torch.zeros((num_images, self.H, self.W, self.N_samples)).to(self.device)
        self.samplePoints = torch.zeros((num_images, self.H, self.W, self.N_samples, 3)).to(self.device)
        self.viewrDirections = torch.zeros((num_images, self.H, self.W, 3)).to(self.device)
        for i in range(num_images):
            #resize image
            resized_img = cv2.resize(images[i],(self.H,self.W))
            self.target_images[i] = np.array(resized_img)
            #ray direction
            origins, directions = self.get_rays(self.H, self.W, focal, poses[i])
            #dists
            z_vals = torch.linspace(self.near, self.far, self.N_samples).expand(origins.shape[:-1] + (self.N_samples,))
            z_vals = z_vals.clone()
            z_vals += torch.rand(list(origins.shape[:-1]) + [self.N_samples]) * (self.far - self.near) / self.N_samples
            dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], 
                        torch.broadcast_to(torch.tensor([1e10], 
                        device=z_vals.device), z_vals[..., :1].shape)], -1)
            self.distances[i] = dists
            #pts
            pts = origins[..., None, :] + directions[..., None, :] * z_vals.to(self.device)[..., :, None]
            pts = pts.reshape(self.H,self.W, self.N_samples, 3)
            self.samplePoints[i] = pts
            #viewer directions 
            viewdirs =  directions / torch.norm(directions, p=2, dim=-1, keepdim=True)
            self.viewrDirections[i] = viewdirs
        self.target_images = torch.from_numpy(self.target_images).cuda()
        
    def get_bbx(self):
        pts = self.samplePoints.reshape(-1,3)
        x_min, y_min, z_min = pts.min(dim=0)[0]
        x_max, y_max, z_max = pts.max(dim=0)[0]
        return (torch.tensor([x_min, y_min, z_min])-torch.tensor([0.1,0.1,0.0001]), 
                torch.tensor([x_max, y_max, z_max])+torch.tensor([0.1,0.1,0.0001]))
    
    def get_rays(self, H, W, focal, c2w):
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=self.device),
                            torch.arange(H, dtype=torch.float32, device=self.device), indexing='xy')
        dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def get_data(self, img_i):
        viewdirs = self.viewrDirections[img_i]
        target_image = self.target_images[img_i]
        dists = self.distances[img_i]
        pts = self.samplePoints[img_i]
        return pts, dists, target_image, viewdirs