import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
from skimage.transform import resize

def save_images(target_images, intermediate_images, result_name, model_name, iterations, psnrs):
    fig, axs = plt.subplots(2, 6, figsize=(10, 6))
    for ax, img in zip(axs[0], target_images):
        ax.imshow(img)
        ax.set_title('Target', fontsize=13)
        ax.axis('off')
    for ax, img, psnr in zip(axs[1], intermediate_images, psnrs):
        ax.imshow(img)
        ax.set_title(f'PSNR: {psnr:.2f}', fontsize=13)
        ax.axis('off')
        
    plt.suptitle(f'{model_name} ({iterations} iterations)', fontsize=16)
    plt.tight_layout() 
    plt.plot()
    plt.savefig('results/'+ result_name)

def save_video(intermediate_images, result_name, target_size=(512, 512)):
    def to8b(x):
        return (255 * np.clip(x, 0, 3)).astype(np.uint8)
    resized_images = [resize(image, target_size, anti_aliasing=True) for image in intermediate_images]
    testsavedir = os.path.join('results', 'videos')
    os.makedirs(testsavedir, exist_ok=True)
    imageio.mimwrite(os.path.join(testsavedir,  result_name + '.mp4'), 
                     to8b(resized_images), fps=30, codec='libx264')