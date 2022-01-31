# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 00:42:55 2022

@author: HP_PC2
"""
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

# Loading and prepare data
tensor_transform = transforms.ToTensor()

dataset = datasets.MNIST(root='/data',
                         train=True,
                         download=True,
                         transform=tensor_transform)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=32,
                                     shuffle=True)

    
# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(in_features=28 * 28, out_features= 128),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=128, out_features=64),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=64, out_features=36),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=36, out_features=18),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=18, out_features=9)
		)
		
		# Building an linear decoder with Linearlayer
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(in_features=9, out_features=18),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=18, out_features=36),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=36, out_features=64),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=64, out_features=128),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=128, out_features=28 * 28),
			torch.nn.Sigmoid()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

        
# Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=1e-1, weight_decay=1e-8)


epochs = 20
outputs = []
losses = []
for epoch in range(epochs):
    for (image, _) in loader:
        # Reshaping the image to (-1, 784)
	    image = image.reshape(-1, 28*28)
		
	    # Output of Autoencoder
	    reconstructed = model(image)
		
	    # Calculating the loss function
	    loss = loss_function(reconstructed, image)
		
	   # The gradients are set to zero,
	   # the the gradient is computed and stored.
	   # .step() performs parameter update
	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()
		
	# Storing the losses in a list for plotting
	    losses.append(loss)
    outputs.append((epochs, image, reconstructed))
	

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')

# Plotting the last 100 values
plt.plot(losses[-100:])


# plot The first input image array and the first reconstructed input image
for i, item in enumerate(image):
    # Reshape the array for plotting
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
    
for i, item in enumerate(reconstructed):
    item = item.reshape(-1, 28, 28)
    plt.imshow(item[0])
















