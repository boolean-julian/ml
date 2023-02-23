import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from MyNeuralNetwork import MyNeuralNetwork

my_model = MyNeuralNetwork()
my_model.load_state_dict(torch.load("my_model.pth"))

device = "cpu"
loss_fn = nn.CrossEntropyLoss()

# Load and test data from MNIST dataset
test_data = datasets.FashionMNIST(
	root = "data",
	train = False,
	download = True,
	transform = ToTensor(),
)

batch_size = 64
test_dataloader = DataLoader(test_data, batch_size = batch_size)

def test(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.eval()

	test_loss, correct = 0, 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	test_loss /= num_batches
	correct /= size

	print(f"Test error:")
	print(f"Accuracy = {100*correct:>0.1f}%")
	print(f"Average loss = {test_loss:>8f}")

test(test_dataloader, my_model, loss_fn)