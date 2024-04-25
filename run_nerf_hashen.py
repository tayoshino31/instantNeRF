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

#hash encoding params
num_levels = 8
min_res = 16
max_res = 128
log2_hashmap_size = 4
features_per_level = 2

#output image
width = 256
height = 256
N_samples = 32

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(0)
I = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
w1 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w2 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w3 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')

b1 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')

hashencoder = encodings.HashEncoding(
    num_levels=num_levels,
    min_res=min_res,
    max_res=max_res,
    log2_hashmap_size=log2_hashmap_size,
    features_per_level=features_per_level,
    hash_init_scale=10.0, #0.001
    implementation="torch",
).to("cuda")
dataset = Dataset()

img_i = 0 #img_i = np.random.randint(100)
x, z_vals, target_image = dataset.get_data(img_i)
feature_grid = hashencoder(x)
feature_grid = torch.tensor(feature_grid, dtype=torch.float, device='cuda:0',requires_grad=True)

optimizer = torch.optim.Adam([w1, w2, w3, b1, b2, b3, feature_grid], lr=3e-2)
loss_fn = torch.nn.MSELoss()

intermediate_images = []
iterations = 5000

import time
start = time.time()

for i in range(iterations):
    # img_i = 0 #img_i = np.random.randint(100)
    # x, z_vals, target_image = dataset.get_data(img_i)
    # feature_grid = hashencoder(x)
    
    y_pred = RenderImage.apply(
        width, height,
        feature_grid, 
        I + w1 * 0.05,
        I + w2 * 0.05,
        I + w3 * 0.05,
        b1, b2, b3)

    y_pred = rendering(y_pred, z_vals)
    loss = loss_fn(y_pred, target_image)
    
    psnr = -10. * torch.log(loss) / torch.math.log(10.)
    print(f"Iteration {i}, Loss: {loss.item()}, psnr: {psnr.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        intermediate_images.append(y_pred.detach().cpu().numpy())

end = time.time()

# Display images side-by-side
import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(intermediate_images[0])
ax2.imshow(y_pred.detach().cpu().numpy())
ax3.imshow(target_image.detach().cpu().numpy())

# Label images
ax1.set_title('Initial')
ax2.set_title('Pred')
ax3.set_title('Target')

plt.show()
plt.savefig('output.png')