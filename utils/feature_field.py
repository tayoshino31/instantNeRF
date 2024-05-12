
# Heavily drew from
# NeRF Studio: https://github.com/nerfstudio-project/nerfstudio/blob/bc9328c7ff70045fce21838122f48ab5201c4ae3/nerfstudio/field_components/encodings.py#L310
# hashNeRF: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/780f4bb8dedb1d5a9af23fae226973e79cbcc48e/run_nerf_helpers.py#L66

import torch
import torch.nn as nn
import math

def normalize_coordinates(x, bounding_box):
    min_xyz = bounding_box[0].to('cuda') 
    max_xyz = bounding_box[1].to('cuda') 
    range_xyz = max_xyz - min_xyz
    x_shape = x.shape
    x = x.reshape(-1,3)
    range_xyz[range_xyz == 0] = 1.0
    x = (x - min_xyz)/range_xyz
    x = x.reshape(x_shape)
    return x

class FeatureField(nn.Module): #hashmap_scale=.0001, res=1024
    def __init__(self, hashmap_scale=1, log2_hashmap_size=19, n_levels = 16, features_per_level=2, base_resolution=16, res=128):
        super().__init__()
        self.hashtable_size = 2**log2_hashmap_size
        self.features_per_level = features_per_level
        self.n_levels = n_levels
        self.max_resolution = torch.tensor(res)
        self.base_resolution = torch.tensor(base_resolution)
        self.growth_factor = torch.exp((torch.log(self.max_resolution)-torch.log(self.base_resolution))/(n_levels-1))
        # Init the hash table (also i think that it should specify device automatically?????)
        self.hashtable = torch.rand(size = (self.hashtable_size * 1, features_per_level), device='cuda:0') * 2 - 1 # table_size * levels * features_per_level
        self.hashtable *= hashmap_scale
        self.hashtable = nn.Parameter(self.hashtable)

    def forward(self, x):
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.growth_factor**i)
            x_embedded = self.encode(x, resolution)
            x_embedded_all.append(x_embedded)
        return torch.cat(x_embedded_all, dim=-1)

    #FOR NOW im assuming x.shape = (3)
    def encode(self, x, res):
        x_reshaped = x.reshape(-1, 3)
        x_scaled = x_reshaped * res

        x_floor = torch.floor(x_scaled).int()
        x_ceil = torch.ceil(x_scaled).int()

        h0 = self.hash(torch.stack((x_floor[ : , 0], x_floor[ : , 1], x_floor[ : , 2]))) #000
        h1 = self.hash(torch.stack((x_ceil[ : , 0], x_floor[ : , 1], x_floor[ : , 2]))) #100
        h2 = self.hash(torch.stack((x_floor[ : , 0], x_ceil[ : , 1], x_floor[ : , 2]))) #010
        h3 = self.hash(torch.stack((x_floor[ : , 0], x_floor[ : , 1], x_ceil[ : , 2]))) #001
        h4 = self.hash(torch.stack((x_ceil[ : , 0], x_ceil[ : , 1], x_floor[ : , 2]))) #110
        h5 = self.hash(torch.stack((x_ceil[ : , 0], x_floor[ : , 1], x_ceil[ : , 2]))) #101
        h6 = self.hash(torch.stack((x_floor[ : , 0], x_ceil[ : , 1], x_ceil[ : , 2]))) #011
        h7 = self.hash(torch.stack((x_ceil[ : , 0], x_ceil[ : , 1], x_ceil[ : , 2]))) #111

        # hash all points
        v0 = self.hashtable[h0]
        v1 = self.hashtable[h1]
        v2 = self.hashtable[h2]
        v3 = self.hashtable[h3]
        v4 = self.hashtable[h4]
        v5 = self.hashtable[h5]
        v6 = self.hashtable[h6]
        v7 = self.hashtable[h7]

        x_difference = x_scaled - x_floor
        encoded = self.trilinear_interpolation(x_difference, v0, v1, v2, v3, v4, v5, v6, v7)

        output_shape = x.shape[:-1] + (-1,)
        return encoded.reshape(output_shape)
                
    #assume 1 point
    def hash(self, x):
        prime0 = 1
        prime1 = 2654435761
        prime2 = 805459861
        result = (prime0 * x[0]) ^ (prime1 * x[1]) ^ (prime2 * x[2])
        return result % self.hashtable_size


    # https://en.wikipedia.org/wiki/Trilinear_interpolation
    def trilinear_interpolation(self, p_d, c_000, c_100, c_010, c_001, c_110, c_101, c_011, c_111):
        #interpolate along x
        # print(p_d[:, 0][:, None])
        c_00 = c_000 * (1-p_d[:, 0][:, None]) + c_100 * p_d[:, 0][:, None]
        c_01 = c_001 * (1-p_d[:, 0][:, None]) + c_101 * p_d[:, 0][:, None]
        c_10 = c_010 * (1-p_d[:, 0][:, None]) + c_110 * p_d[:, 0][:, None]
        c_11 = c_011 * (1-p_d[:, 0][:, None]) + c_111 * p_d[:, 0][:, None] # c_11 = c_010 * (1-p_d[:, 0][:, None]) + c_111 * p_d[:, 0][:, None]

        #interpolate along y
        c_0 = c_00 * (1-p_d[:, 1][:, None]) + c_10 * p_d[:, 1][:, None]
        c_1 = c_01 * (1-p_d[:, 1][:, None]) + c_11 * p_d[:, 1][:, None]

        #interpolate along z
        c = c_0 * (1-p_d[:, 2][:, None]) + c_1 * p_d[:, 2][:, None]

        return c
   