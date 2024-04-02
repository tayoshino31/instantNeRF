# import torch
# from sampling import Dataset
# import os
# import matplotlib.pyplot as plt
# import time
# from volume_rendering import rendering
# from nerfstudio.field_components import encodings
# from nn import NeuralNetwork
# import numpy as np

# #hash encoding params
# num_levels = 8
# min_res = 16
# max_res = 128
# log2_hashmap_size = 4
# features_per_level = 3

# #output image
# width = 128
# height = 128
# N_samples = 32

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# torch.manual_seed(0)
# hashencoder = encodings.HashEncoding(
#     num_levels=num_levels,
#     min_res=min_res,
#     max_res=max_res,
#     log2_hashmap_size=log2_hashmap_size,
#     features_per_level=features_per_level,
#     hash_init_scale= 10.0,    # 0.001
#     implementation="torch",
# ).to("cuda")

# dataset = Dataset()
# model = NeuralNetwork(hash_encoding=hashencoder)
# model.to("cuda")

# optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = torch.nn.MSELoss()

# intermediate_images = []
# iterations = 1000

# import time
# start = time.time()

# for i in range(iterations):
#     img_i = 0 #img_i = np.random.randint(100)
#     x, z_vals, target_image = dataset.get_data(img_i)
#     y_pred = model(x)
#     y_pred = y_pred.reshape(128,128,32,4)
#     y_pred = rendering(y_pred, z_vals)
    
#     loss = loss_fn(y_pred, target_image)
#     psnr = -10. * torch.log(loss) / torch.math.log(10.)
#     print(f"Iteration {i}, Loss: {loss.item()}, psnr: {psnr.item()}")

#     optimizer_model.zero_grad()
    
#     loss.backward()
#     optimizer_model.step()
    
#     if i % 20 == 0:
#         intermediate_images.append(y_pred.detach().cpu().numpy())

# end = time.time()

# # Display images side-by-side
# import matplotlib.pyplot as plt
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(intermediate_images[0])
# ax2.imshow(y_pred.detach().cpu().numpy())
# ax3.imshow(target_image.detach().cpu().numpy())

# # Label images
# ax1.set_title('Initial')
# ax2.set_title('Pred')
# ax3.set_title('Target')

# plt.show()
# plt.savefig('output.png')


import torch
from sampling import Dataset
import os
import matplotlib.pyplot as plt
import time
from volume_rendering import rendering
from nerfstudio.field_components import encodings
from nn import NeuralNetwork
import numpy as np
from hash_encoder import HashEmbedder

#hash encoding params
num_levels = 8
min_res = 16
max_res = 128
log2_hashmap_size = 2
features_per_level = 2

#output image
width = 128
height = 128
N_samples = 32

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(0)
model = NeuralNetwork()
dataset = Dataset()
model.to("cuda")


bounding_box = dataset.get_bbox3d_for_blenderobj()
hashencoder = HashEmbedder(bounding_box=bounding_box)

# hashencoder = encodings.HashEncoding(
#     num_levels=num_levels,
#     min_res=min_res,
#     max_res=max_res,
#     log2_hashmap_size=log2_hashmap_size,
#     features_per_level=features_per_level,
#     hash_init_scale= 10.0,    # 0.001
#     implementation="torch",
# ).to("cuda")


# img_i = 0 #img_i = np.random.randint(100)
# x, z_vals, target_image = dataset.get_data(img_i)
# feature_grid = hashencoder(x)
# feature_grid = torch.tensor(feature_grid, dtype=torch.float, device='cuda:0',requires_grad=True)

optimizer_model = torch.optim.Adam(model.parameters(), lr=3e-3)
#optimizer_hash = torch.optim.Adam([feature_grid], lr=2e-3)
optimizer_hash = torch.optim.Adam(list(hashencoder.parameters()), lr=3e-3)
loss_fn = torch.nn.MSELoss()

intermediate_images = []
iterations = 500

import time
start = time.time()

for i in range(iterations):
    img_i = 0 #img_i = np.random.randint(100)
    x, z_vals, target_image = dataset.get_data(img_i)
    feature_grid = hashencoder(x)
    
    feature_grid = feature_grid[0].reshape(-1,16) * 10000
    y_pred = model(feature_grid)
    y_pred = y_pred.reshape(128,128,32,4)
    
    y_pred = rendering(y_pred, z_vals)
    
    loss = loss_fn(y_pred, target_image)
    psnr = -10. * torch.log(loss) / torch.math.log(10.)
    print(f"Iteration {i}, Loss: {loss.item()}, psnr: {psnr.item()}")

    optimizer_model.zero_grad()
    optimizer_hash.zero_grad()
    
    loss.backward()
    optimizer_model.step()
    optimizer_hash.step()
    
    if i % 100 == 0:
        intermediate_images.append(y_pred.detach().cpu().numpy())

end = time.time()

import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
ax1.imshow(intermediate_images[0])
ax2.imshow(intermediate_images[1])
ax3.imshow(intermediate_images[2])
ax4.imshow(intermediate_images[3])
ax5.imshow(intermediate_images[4])
ax6.imshow(target_image.detach().cpu().numpy())

# Label images
ax1.set_title('Initial')
#ax2.set_title(f'Iter({iterations} psnr {intermediate_psnr[0]})')
ax6.set_title('Target')

plt.show()
plt.savefig('output.png')

# # Display images side-by-side
# import matplotlib.pyplot as plt
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(intermediate_images[0])
# ax2.imshow(y_pred.detach().cpu().numpy())
# ax3.imshow(target_image.detach().cpu().numpy())

# # Label images
# ax1.set_title('Initial')
# ax2.set_title('Pred')
# ax3.set_title('Target')

# plt.show()
# plt.savefig('output.png')