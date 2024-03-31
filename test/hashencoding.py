#https://docs.nerf.studio/nerfology/model_components/visualize_encoders.html
import torch
from nerfstudio.field_components import encodings as encoding

num_levels = 8
min_res = 2
max_res = 128
log2_hashmap_size = 4  # Typically much larger tables are used

resolution = 128
slice = 0

# Fixing features_per_level to 3 for easy RGB visualization. Typical value is 2 in networks
features_per_level = 3

encoder = encoding.HashEncoding(
    num_levels=num_levels,
    min_res=min_res,
    max_res=max_res,
    log2_hashmap_size=log2_hashmap_size,
    features_per_level=features_per_level,
    hash_init_scale=0.001,
    implementation="torch",
)

x_samples = torch.linspace(0, 1, resolution)
grid = torch.stack(torch.meshgrid([x_samples, x_samples, x_samples], indexing="ij"), dim=-1)

encoded_values = encoder(grid)
print(grid.shape)
print(encoded_values.shape)
print(encoder.parameters())