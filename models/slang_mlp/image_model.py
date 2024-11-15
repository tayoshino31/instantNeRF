import slangpy
import torch
import os
#from torch.profiler import profile, record_function, ProfilerActivity

launchBlockSize = (32, 4, 1)

# Before loading our module, we're going to pre-process the TORCH_CUDA_ARCH_LIST tag from the environment. 
# This is to get around a bug with how torch constructs arch flags: explicitly providing arch flags seems to
# not override torch's default arch flags and cause compiler errors due to setting the SM version too low.
# 
# We therefore replace the env list with a restricted list of arches that we know will work.
#
if "TORCH_CUDA_ARCH_LIST" in os.environ:
    # Go through space-separated list of arches and remove any that are below 8.0
    arches = os.environ["TORCH_CUDA_ARCH_LIST"].split(" ")
    arches = [arch for arch in arches if not int(arch.split(".")[0]) <= 7]
    os.environ["TORCH_CUDA_ARCH_LIST"] = " ".join(arches)

m = slangpy.loadModule('image-model.slang', 
                       defines={
                            'NUM_THREADS_PER_BLOCK': launchBlockSize[0] * launchBlockSize[1] * launchBlockSize[2],
                            'WARP_SIZE': 32})
class RenderImage(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, width, height, feature_grid, viewdirs, dists, embedded, *args):
        weights = args[0: 3]
        biases = args[3: 6]
        output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        linear_layers = [m.Linear(weights=weights[i], bias=biases[i]) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)
        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], 
                    (height + blockSize[1] - 1) // blockSize[1], 1)
        m.renderImage(mlp=mlp, featureGrid=feature_grid,
                      viewDir = viewdirs, dists = dists,
                      imageOutput=output, embedded = embedded).launchRaw(blockSize=blockSize, gridSize=gridSize)
        ctx.save_for_backward(output, feature_grid, viewdirs, dists, embedded, *args)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, feature_grid, viewdir, dists, embedded, *args = ctx.saved_tensors
        weights = args[0: 3]
        biases = args[3: 6]
        weights_d = [torch.zeros_like(w) for w in weights]
        biases_d = [torch.zeros_like(b) for b in biases]
        feature_grid_d = torch.zeros_like(feature_grid)
        viewdir_d = torch.zeros_like(viewdir)
        dists_d = torch.zeros_like(dists)
        width, height, _ = output.shape
        
        linear_layers = [m.Linear(weights=(weights[i], weights_d[i]), bias=(biases[i], biases_d[i])) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)
        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], 
                    (height + blockSize[1] - 1) // blockSize[1], 1)
        m.renderImage.bwd(mlp=mlp, featureGrid=(feature_grid, feature_grid_d), 
                        viewDir=(viewdir, viewdir_d), dists = (dists, dists_d),
                        imageOutput=(output, grad_output), embedded = embedded).launchRaw(blockSize=blockSize, gridSize=gridSize)
        return None, None, feature_grid_d, None, None, None, *weights_d, *biases_d