import numpy as np
import torch
import cv2

class Dataset():
    def __init__(self):
        self.data = np.load('../../nerf_dataset/dataset.npz')
        self.images = self.data['images']
        self.poses = self.data['poses']
        self.focal = self.data['focal']
        self.device = "cuda"
        self.H, self.W = 128, 128 #64, 64 

        self.testimg,  self.testpose = self.images[101], self.poses[101]
        self.images = self.images[:100,...,:3]
        self.poses = self.poses[:100]
        self.near= 2
        self.far = 6
        self.N_samples = 32

    def get_rays(self, H, W, focal, c2w):
        device = "cuda"
        i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device),
                            torch.arange(H, dtype=torch.float32, device=device), indexing='xy')
        dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d
    
    def get_bbox3d_for_blenderobj(self, near=2.0, far=6.0):
        #focal = 0.5*W/np.tan(0.5 * camera_angle_x)

        # ray directions in camera coordinates
       #directions = get_ray_directions(H, W, focal)

        min_bound = [100, 100, 100]
        max_bound = [-100, -100, -100]
    
        #points = []
        for pose in self.poses:
            pose = torch.tensor(pose, dtype=torch.float32, device=self.device)
            focal = torch.tensor(self.focal, dtype=torch.float32, device=self.device).clone().detach()
            rays_o, rays_d = self.get_rays(self.H, self.W, focal, pose)

            min_point = rays_o + near*rays_d
            max_point = rays_o + far*rays_d
            min_point = min_point.view(-1, 3).min(dim=0)[0]
            max_point = max_point.view(-1, 3).max(dim=0)[0]
            
            for i in range(3):
                if(min_bound[i] > min_point[i]):
                    min_bound[i] = min_point[i]
                if(max_bound[i] < min_point[i]):
                    max_bound[i] = min_point[i]
                

        return ((torch.tensor(min_bound)-torch.tensor([1.0,1.0,1.0])).to(self.device), 
                (torch.tensor(max_bound)+torch.tensor([1.0,1.0,1.0])).to(self.device))


    def get_data(self, img_i):
        img64 = cv2.resize(self.images[img_i],(self.H,self.W))
        img64 = np.array(img64)
        testimg = torch.from_numpy(img64).cuda()
        
        target_image = testimg.type(torch.float)
        pose = torch.tensor(self.poses[img_i], dtype=torch.float32, device=self.device)
        focal = torch.tensor(self.focal, dtype=torch.float32, device=self.device).clone().detach()
        rays_o, rays_d = self.get_rays(self.H, self.W, focal, pose)
        
        z_vals = torch.linspace(self.near, self.far, self.N_samples, 
                                device=rays_o.device).expand(rays_o.shape[:-1] + (self.N_samples,))
        z_vals = z_vals.clone()
        z_vals += torch.rand(list(rays_o.shape[:-1]) + [self.N_samples], 
                            device=rays_o.device) * (self.far - self.near) / self.N_samples
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(self.H,self.W, self.N_samples, 3)
        return pts, z_vals, target_image