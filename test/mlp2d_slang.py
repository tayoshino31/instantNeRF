import torch
import slangpy

launchBlockSize = (32, 8, 1)
m = slangpy.loadModule('../image-model.slang', 
                       defines={
    'NUM_THREADS_PER_BLOCK': launchBlockSize[0] * launchBlockSize[1] * launchBlockSize[2],
    'WARP_SIZE': 32})
C = 16
H = 1
W = 1

torch.manual_seed(0)
torch.cuda.manual_seed(0)
X = torch.randn((H, W , C), dtype = torch.float, device='cuda')

w1 = torch.randn((C,C), dtype = torch.float,  requires_grad=True, device='cuda')
w2 = torch.randn((C,C), dtype = torch.float,  requires_grad=True, device='cuda')
w3 = torch.randn((C,C), dtype = torch.float,  requires_grad=True, device='cuda')

b1 = torch.randn(C, dtype = torch.float,  requires_grad=True, device='cuda')
b2 = torch.randn(C, dtype = torch.float,  requires_grad=True, device='cuda')
b3 = torch.randn(C, dtype = torch.float,  requires_grad=True, device='cuda')

optimizer = torch.optim.SGD([w1, w2, w3, b1, b2, b3], lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()


class RenderImage(torch.autograd.Function):
    def forward(ctx, width, height, feature_grid, *args):
        weights = args[0: 3]
        biases = args[3: 6]
        #output = torch.zeros((width, height, 3), dtype=torch.float).cuda()
        output = torch.zeros((width, height, 16), dtype=torch.float).cuda()
        
        linear_layers = [m.Linear(weights=weights[i], bias=biases[i]) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage(mlp=mlp, featureGrid=feature_grid, imageOutput=output).launchRaw(blockSize=blockSize, gridSize=gridSize)

        ctx.save_for_backward(output, feature_grid, *args)

        return output
    
    def backward(ctx, grad_output):
        output, feature_grid, *args = ctx.saved_tensors
        weights = args[0: 3]
        biases = args[3: 6]

        weights_d = [torch.zeros_like(w) for w in weights]
        biases_d = [torch.zeros_like(b) for b in biases]
        feature_grid_d = torch.zeros_like(feature_grid)

        width, height, _ = output.shape
        
        linear_layers = [m.Linear(weights=(weights[i], weights_d[i]), bias=(biases[i], biases_d[i])) for i in range(3)]
        mlp = m.MLP(layers=linear_layers)

        blockSize = launchBlockSize
        gridSize = ((width + blockSize[0] - 1) // blockSize[0], (height + blockSize[1] - 1) // blockSize[1], 1)

        m.renderImage.bwd(mlp=mlp, featureGrid=(feature_grid, feature_grid_d), imageOutput=(output, grad_output)).launchRaw(blockSize=blockSize, gridSize=gridSize)

        return None, None, feature_grid_d, *weights_d, *biases_d

x = X
y_pred = RenderImage.apply(W, H, x, w1, w2, w3, b1, b2, b3)
print("MLP2 forward")
print(y_pred)
loss = y_pred.sum()
loss.backward()
print("MLP2 backward")

#print(x.grad.cpu())

# print("w_grad")
# print(w1.grad.cpu())
# print(w2.grad.cpu())
# print(w3.grad.cpu())
# print("b_grad")
# print(b1.grad.cpu())
# print(b2.grad.cpu())
# print(b3.grad.cpu())