import matplotlib.pyplot as plt
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