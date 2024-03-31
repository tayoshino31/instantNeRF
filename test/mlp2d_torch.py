import torch
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

print("expected output")

x = X
#x = X[0,:].unsqueeze(0)
out1 = x @ w1 + b1
out1 = torch.relu(out1)
out2 = out1 @ w2 + b2
out2 = torch.relu(out2)
out3 = out2 @ w3 + b3
out3 = torch.relu(out3)
print("expected forward")
print(out3)
# result_loss = out3.sum()
# result_loss.backward()
# print("expected backward")
# #print(x.grad.cpu())
# print("w_grad")
# print(w1.grad.cpu())
# print(w2.grad.cpu())
# print(w3.grad.cpu())
# print("b_grad")
# print(b1.grad.cpu())
# print(b2.grad.cpu())
# print(b3.grad.cpu())