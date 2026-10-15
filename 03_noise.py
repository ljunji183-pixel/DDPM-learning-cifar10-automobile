import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
#running on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data pixel values are in the range [-1, 1],nomalization

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.CIFAR10(root=r"D:\PythonProject\diffusion-model\my-ddpm\data", train=True, download=False, transform=transform)
x_0, _ = dataset[20]  # Get the 20th image and its label
x_0 = x_0.unsqueeze(0).to(device)  # Move the image to the GPU if available

# ddpm(noise schedule) parameters
T = 1000  # Total number of timesteps
betas = torch.linspace(1e-4, 0.02, T).to(device)  # Linear noise schedule
alphas = 1 - betas
a_bars = torch.cumprod(alphas, dim=0)  # Cumulative product of (1 - beta_t)
#define add noise function  q(x_t | x_0) = N(x_t; sqrt(a_bar_t) * x_0, (1 - a_bar_t) * I)
def q_sample(x_0, t,noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).to(device) #noise sa,mple from standard normal distribution
    a_bar_t = a_bars[t].view(-1, 1, 1, 1)  # Get the cumulative product for the given timestep
    return torch.sqrt(a_bar_t) * x_0 + torch.sqrt(1 - a_bar_t) * noise  # Return the noisy image

#catch each timestep and add noise to the image

timesteps = [0, 50, 100, 200, 500, 999]
noisy_images = []
for step in timesteps:
    t=torch.tensor(step, dtype=torch.long).to(device)
    noisy_image = q_sample(x_0, t)
    noisy_image = (noisy_image.clamp(-1, 1)+1)/2  # Clamp the values to be in the range [-1, 1]
    noisy_images.append(noisy_image)

#save row of noisy images
noisy_row = torch.cat(noisy_images, dim=0)  
save_image(noisy_row, "noisy_images.png")
print("Noisy images saved as: noisy_images.png")