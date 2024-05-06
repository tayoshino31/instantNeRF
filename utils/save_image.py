import matplotlib.pyplot as plt
def save_images(target_images, intermediate_images, name):
    fig, axs = plt.subplots(2, 6, figsize=(10, 6))
    for ax, img in zip(axs[0], target_images):
        ax.imshow(img)
        ax.set_title('Target')
        ax.axis('off')
    for ax, img in zip(axs[1], intermediate_images):
        ax.imshow(img)
        ax.set_title('Pred')
        ax.axis('off')
    plt.tight_layout() 
    plt.plot()
    plt.savefig('results/'+ name)