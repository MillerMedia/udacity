# Imports here
import torch
import json
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch import nn, optim, utils
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

"""
Helper Functions
"""
class DenseNet_Helper():
	def __init__(self, hidden_units=512, classifier=None):
		self.name = 'DenseNet'
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
		) if classifier is None else classifier

	def set_classifier(self, model):
		model.classifier = self.classifier
		return model

	def get_optimizer_parameters(self, model):
		return model.classifier.parameters()

	def get_state_dict(self, model):
		return model.classifier.state_dict()


class ResNet_Helper():
	def __init__(self, hidden_units=512, classifier=None):
		self.name = 'ResNet'
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
		) if classifier is None else classifier

	def set_classifier(self, model):
		model.fc = self.classifier
		return model

	def get_optimizer_parameters(self, model):
		return model.fc.parameters()

	def get_state_dict(self, model):
		return model.fc.state_dict()

def load_checkpoint(filepath, gpu=False):
	device = "cuda:0" if torch.cuda.is_available() and gpu==True else "cpu"
	checkpoint = torch.load(filepath, map_location=device)

	model_helper = False
	if 'densenet' in checkpoint['model_name']:
		model_helper = DenseNet_Helper(classifier=checkpoint['classifier'])
	elif 'resnet' in checkpoint['model_name']:
		model_helper = ResNet_Helper(classifier=checkpoint['classifier'])

	if not model_helper:
		raise('Model name ' + str(checkpoint['model_name']) + ' is not valid. Aborting.')

	model = model_helper.model

	# Freeze parameters since we are using a pre-trained network and do not need back propagation
	for param in model.parameters():
		param.requires_grad = False

	# Load the state dict
	# Loop taken from: https://discuss.pytorch.org/t/transfer-learning-missing-key-s-in-state-dict-unexpected-key-s-in-state-dict/33264/3
	state_dict = checkpoint['state_dict']
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]  # remove 'module.' of DataParallel
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict, strict=False)
	model = model.set_classifer(model)

	optimizer = optim.Adam(model.get_optimizer_parameters(model), lr=checkpoint['learning_rate'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	return model, optimizer, checkpoint['class_to_idx']

parser = ArgumentParser()
parser.add_argument('input',
					help='Path to image to be analyzed.')
parser.add_argument('checkpoint',
					help='Path to checkpoint file.')
parser.add_argument('--top_k',
					help='Number of top predicted classes to output.')
parser.add_argument('--category_names',
					help='Category mapping. Format should be JSON.')
parser.add_argument("--gpu", dest="gpu", default=False,
					help="Use GPU for training", action="store_true")

args = parser.parse_args()

# Parse command line arguments
path_to_image = args.path_to_image
top_k = args.top_k

with open(args.category_names, 'r') as f:
    category_names = json.load(f)

gpu_available = args.gpu