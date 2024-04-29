import torch
from sampling import Dataset
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time
from volume_rendering import rendering
from encoding import embed_fn
from image_model import RenderImage
from nerfstudio.field_components import encodings
import numpy as np

#hash encoding params
num_levels = 8
min_res = 16
max_res = 128
log2_hashmap_size = 4
features_per_level = 3

#output image
width = 256
height = 256
N_samples = 32

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
C = 32
torch.manual_seed(0)
I = torch.diag(torch.ones(C, dtype=torch.float)).cuda().contiguous()
w1 = torch.randn((C, C), dtype=torch.float, requires_grad=True, device='cuda:0')
w2 = torch.randn((C, C), dtype=torch.float, requires_grad=True, device='cuda:0')
w3 = torch.randn((C, C), dtype=torch.float, requires_grad=True, device='cuda:0')

b1 = torch.zeros(C, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(C, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(C, dtype=torch.float, requires_grad=True, device='cuda:0')

dataset = Dataset()
optimizer = torch.optim.Adam([w1, w2, w3, b1, b2, b3], lr=5e-3) 
loss_fn = torch.nn.MSELoss()

intermediate_images = []
target_images = []
iterations = 2500

import time
start = time.time()

for i in range(iterations):
    img_i = 0  #img_i = np.random.randint(100)
    x, z_vals, target_image, viewdirs = dataset.get_data(img_i)
    encoded_x = x                #encoded_x = embed_fn(x)
    encoded_viewdirs = viewdirs  #encoded_viewdirs = embed_fn(viewdirs)

    y_pred = RenderImage.apply(
        width, height,
        encoded_x, encoded_viewdirs,
        w1, w2, w3,
        b1, b2, b3)
    y_pred = rendering(y_pred, z_vals)
    loss = loss_fn(y_pred, target_image)
    
    psnr = -10. * torch.log(loss) / torch.math.log(10.)
    print(f"Iteration {i}, Loss: {loss.item()}, psnr: {psnr.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i  > (iterations - 5):
        intermediate_images.append(y_pred.detach().cpu().numpy())
        target_images.append(target_image.detach().cpu().numpy())

end = time.time()


#Multiple images
# fig, axs = plt.subplots(2, 4, figsize=(10, 6))
# for ax, img in zip(axs[0], target_images):
#     ax.imshow(img)
#     ax.set_title('Target')
#     ax.axis('off')
# for ax, img in zip(axs[1], intermediate_images):
#     ax.imshow(img)
#     ax.set_title('Pred')
#     ax.axis('off')
# plt.tight_layout() 
# plt.show() 
# plt.savefig('output_multiple.png')


# #Interatively train a single image
fig, (ax2, ax3) = plt.subplots(1, 2)
ax2.imshow(y_pred.detach().cpu().numpy())
ax3.imshow(target_image.detach().cpu().numpy())
ax2.set_title(f"Pred (PSNR:{psnr.item():.2f})")
ax3.set_title('Target')
plt.show()
plt.savefig('output_single.png')