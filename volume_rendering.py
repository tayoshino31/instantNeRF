import torch

def rendering(x, z_vals):
    sigma_a = torch.relu(x[..., 3])
    rgb = torch.sigmoid(x[..., :3])
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], 
                        torch.broadcast_to(torch.tensor([1e10], 
                        device=z_vals.device), z_vals[..., :1].shape)], -1)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    padded_alpha = torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1)
    cumprod = torch.cumprod(padded_alpha, dim=-1)[..., :-1] 
    weights_volume = alpha * cumprod
    rgb_pred = torch.sum(weights_volume[..., None] * rgb, -2)
    return rgb_pred