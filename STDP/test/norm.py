import torch

# Set print options
torch.set_printoptions(linewidth=2048, precision=6, sci_mode=False)
# Define a tensor
a = torch.tensor(
    [-1.8297, -1.9489, -1.9937, -1.9937, -2.0225, -2.0375, -2.0499,
     -2.6950, -2.6967, -2.8939, -2.9030, -2.9107, -2.9385, -2.9468,
     -2.9705, -2.9777])
a = torch.sort(input=a, descending=False).values
print(a, torch.mean(a), torch.std(a))

# Z-score standardization
mean_a = torch.mean(a)
std_a = torch.std(a)
n1 = (a - mean_a) / std_a
print(n1, torch.mean(n1), torch.std(n1))

# Min-Max scaling
min_a = torch.min(a)
max_a = torch.max(a)
n2 = (a - min_a) / max_a
print(n2, torch.mean(n2), torch.std(n2))