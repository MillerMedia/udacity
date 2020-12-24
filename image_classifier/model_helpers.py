from torchvision import models
from torch import nn

class DenseNet_Helper():
	def __init__(self, hidden_units=512):
		self.model = models.densenet121(pretrained=True)
		self.classifier = nn.Sequential(
			nn.Linear(1024, hidden_units),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_units, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 102),  # 102 labels returned from CSV
			nn.LogSoftmax(dim=1)
		)

class ResNet_Helper():
	def __init__(self, hidden_units = 512):
		self.model = models.resnet18(pretrained=True)
		self.classifier = nn.Sequential(
			nn.Linear(512, hidden_units),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(hidden_units, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 102),  # 102 labels returned from CSV
			nn.LogSoftmax(dim=1)
		)