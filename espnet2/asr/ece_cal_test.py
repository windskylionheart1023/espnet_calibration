from asr.ece_kde import *

# # Generate dummy probability scores and labels
# f = torch.rand((50, 3), requires_grad=True)
# f = f / torch.sum(f, dim=1).unsqueeze(-1)
# y = torch.randint(0, 3, (50,))

bandwidth = get_bandwidth(f, device='cpu')

loss_ece = get_ece_kde(f, y, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu')

print(loss_ece)