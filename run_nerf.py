import torch
from sampling import Dataset
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import time
from volume_rendering import rendering
from image_model import RenderImage
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(0)
I = torch.diag(torch.ones(16, dtype=torch.float)).cuda().contiguous()
w1 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w2 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')
w3 = torch.randn((16, 16), dtype=torch.float, requires_grad=True, device='cuda:0')

b1 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b2 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')
b3 = torch.zeros(16, dtype=torch.float, requires_grad=True, device='cuda:0')


# Load a target image as a torch tensor
# target_image = torch.from_numpy(plt.imread('sample.jpg')).cuda()[:, :, :3].contiguous()
# target_image = target_image.type(torch.float) / 255.0

width = 128
height = 128
N_samples = 32

dataset = Dataset()
img_i = 0
feature_grid, z_vals, target_image = dataset.get_data(img_i)

feature_grid = torch.randn((width, height, N_samples, 16), 
                             dtype=torch.float, requires_grad=True, device='cuda:0')

# target_image = target_image.permute(2, 0, 1)
# save_image(target_image.cpu(), 'sample.jpg')

# Setup optimization loop
optimizer = torch.optim.Adam([w1, w2, w3, b1, b2, b3, feature_grid], lr=3e-2)
loss_fn = torch.nn.MSELoss()


intermediate_images = []
iterations = 2000

import time
start = time.time()

for i in range(iterations):
    #img_i = 0 #img_i = np.random.randint(height)
    #x, z_vals, target_image = dataset.get_data(img_i)
    
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
ax2.set_title(f'Optimized ({iterations} iterations in {end - start:.2f} seconds)')
ax3.set_title('Target')

plt.show()
plt.savefig('output.png')