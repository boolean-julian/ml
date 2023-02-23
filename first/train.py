import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from MyNeuralNetwork import MyNeuralNetwork

# Load and training data from MNIST dataset
training_data = datasets.FashionMNIST(
	root = "data",
	train = True,
	download = True,
	transform = ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size = batch_size)

for X, y in train_dataloader:
	print("Shape of X: ", X.shape)
	print("Shape of y: ", y.shape)
	break

# Get device for training
device = "cpu"

# Define model
my_model = MyNeuralNetwork().to(device)
print(my_model)

# Optimize model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()

	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		# Compute prediction error
		pred = model(X)
		loss = loss_fn(pred, y)

		# Back propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), (batch+1)*len(X)
			print(f"loss = {loss:>7f}\t[{current:>5d}/{size:>5d}]")

# Train
epochs = 100
for t in range(epochs):
	print("\nEpoch", t+1)
	train(train_dataloader, my_model, loss_fn, optimizer)

filename = "my_model.pth"
torch.save(my_model.state_dict(), filename)
print("Done. Saved PyTorch model state to", filename)