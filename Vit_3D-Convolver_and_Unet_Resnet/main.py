import argparse
import torch
import yaml
from dataloader import get_loader
import os
from sklearn.model_selection import train_test_split
from solver import Solver
# from solver_ae import Solver

def getpaths(config, mode=None):
	if mode == 'train':
		imagefolder = config['input_image_data']['path']
		pulsefolder = config['input_pulse_data']['path']
		imagepulse_paths = []
		for subfolder in list(sorted(os.listdir(imagefolder))):
			imagepulse_paths.append([os.path.join(imagefolder, subfolder), os.path.join(pulsefolder, subfolder)])
		train, val = train_test_split(imagepulse_paths, test_size=0.2, random_state=69)
		return train, val
	else:
		imagefolder = config['input_image_test_data']['path']
		pulsefolder = config['input_pulse_test_data']['path']
		imagepulse_paths = []
		for subfolder in list(sorted(os.listdir(imagefolder))):
			imagepulse_paths.append([os.path.join(imagefolder, subfolder), os.path.join(pulsefolder, subfolder)])
		return imagepulse_paths

def main(args):
	# Open the config file
	with open(args.config_path, 'r') as file:
		config = yaml.safe_load(file)
	# Set the device
	device='cuda' if torch.cuda.is_available() else 'cpu'
	print(device)
	# Creat a train valid split based on runs. keep a separate test set. Later we can vary it.
	trainfol_path, valfol_path = getpaths(config, mode='train')
	testfol_path = getpaths(config, mode='test')
	# Get the dataloader for each of the train valid and test sets
	train_dataset, train_loader = get_loader(config, trainfol_path, mode='train')
	val_dataset, val_loader = get_loader(config, valfol_path, mode='valid')
	test_dataset, test_loader = get_loader(config, testfol_path, mode='test')
	print(f"Length of train data: {len(train_dataset)}, Length of validation data: {len(val_dataset)}, Length of test data: {len(test_dataset)}")
	# Set the training parameters
	solver = Solver(config, train_loader, val_loader, test_loader, device)
	# Create Intermediate Dataset
	# solver.create_dataset_pls('../IntTemporalData', solver.train_loader)
	# solver.create_dataset_pls('../IntTemporalData', solver.valid_loader)
	# solver.create_dataset_pls('../IntTemporalTestData', solver.test_loader)
	# Train the network
	solver.train()
	# Test the network
	# solver.test()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_path', type=str, default='config.yaml', help='Set all configs in the file')
	args = parser.parse_args()
	main(args)