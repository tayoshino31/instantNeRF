import torch
import math
class HashEncoder():
    def __init__(self, num_levels=16, log2_hashmap_size=19, features_per_level=2, min_res=16, max_res=1024):
        self.num_levels = num_levels
        self.hashtable_size = 2**log2_hashmap_size
        self.features_per_level = features_per_level
        self.min_res = min_res
        self.max_res = max_res

        
        #calculate the resolution for each level
        levels = torch.arange(1, num_levels+1)
        scaling = torch.exp(levels * (math.log(max_res) - math.log(min_res))/(num_levels - 1 if num_levels > 1 else 1)) # growth_factor ^ level
        self.resolutions = torch.floor(min_res * scaling)


    #FOR NOW im assuming x.shape = (3)
    def encode(self, x):
        x_scaled = torch.reshape(self.resolutions, (-1, 1)) * x.repeat(self.num_levels, 1) 
        x_floor = torch.floor(x_scaled)
        x_ceil = torch.ceil(x_scaled)
        p0 = torch.stack((x_floor[..., 0], x_floor[..., 1], x_floor[..., 2]), dim=1)
        p1 = torch.stack((x_ceil[..., 0], x_floor[..., 1], x_floor[..., 2]), dim=1)
        p2 = torch.stack((x_floor[..., 0], x_ceil[..., 1], x_floor[..., 2]), dim=1)
        p3 = torch.stack((x_floor[..., 0], x_floor[..., 1], x_ceil[..., 2]), dim=1)
        p4 = torch.stack((x_ceil[..., 0], x_ceil[..., 1], x_floor[..., 2]), dim=1)
        p5 = torch.stack((x_floor[..., 0], x_ceil[..., 1], x_ceil[..., 2]), dim=1)
        p6 = torch.stack((x_ceil[..., 0], x_floor[..., 1], x_ceil[..., 2]), dim=1)
        p7 = torch.stack((x_ceil[..., 0], x_ceil[..., 1], x_ceil[..., 2]), dim=1)
        self.hash(p1)
        
        #for each level
            #x_scaled =  x * gridResolution[level]
            #x_floor
            #x_ceil


            #if course level where (N_l +1^d <= T) the mapping is 1:1
            #else 
                #hash(x_floor)
                #hash(x_ceil)
                
    #assume 1 point
    def hash(self, x):
        prime0 = 1
        prime1 = 2654435761
        prime2 = 805459861
        result = (prime0 * x[0]) ^ (prime1 * x[1]) ^ (prime2 * x[2])
        print("hi")
        return 0 
        #return result % self.hashtable_size