import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.gridspec as gridspec
import re
from collections import defaultdict

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytorch_msssim

master_folder = "../IntIntSpatialData"


class VDSR(nn.Module):
    def __init__(self, num_channels=1):
        super(VDSR, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            *[
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                )
                for _ in range(18)
            ],
            nn.Conv2d(64, num_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        residual = self.layers(x)
        return x + residual

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Patch extraction and representation
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        # Reconstruction
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class LaplacianPyramidLoss(nn.Module):
    def __init__(self, kernel_size=3):
        super(LaplacianPyramidLoss, self).__init__()

        # Define a Laplacian filter kernel (2D kernel for grayscale images)
        # This is a fixed Laplacian kernel
        laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).to('cuda')
        laplacian_kernel = laplacian_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

        # Register the kernel as a buffer so it's not a learnable parameter
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def laplacian_filter(self, image):
        # Apply the Laplacian filter using 2D convolution
        filtered_image = F.conv2d(image, self.laplacian_kernel, padding=1)
        return filtered_image

    def forward(self, pred, target):
        # pred = pred.unsqueeze(1)
        # target = target.unsqueeze(1)
        # Apply Laplacian filter to both predicted and target images
        pred_laplacian = self.laplacian_filter(pred)
        target_laplacian = self.laplacian_filter(target)

        # Compute mean squared error loss between the Laplacian-filtered images
        loss = F.mse_loss(pred_laplacian, target_laplacian)
        return loss


# Sobel Filter implementation for Edge Detection
class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Sobel kernels for edge detection
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

        # Convert to PyTorch tensors and expand dimensions to act as convolutional kernels
        self.sobel_x = torch.Tensor(sobel_x).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.Tensor(sobel_y).unsqueeze(0).unsqueeze(0)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()

    def forward(self, img):
        # img should have shape [batch_size, channels, height, width]
        # Apply Sobel filters for each channel independently
        edges_x = F.conv2d(img, self.sobel_x, padding=1)
        edges_y = F.conv2d(img, self.sobel_y, padding=1)

        # Magnitude of gradients (combined x and y direction edges)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_filter = SobelFilter()
        self.criterion = nn.L1Loss()  # L1 Loss to compare edges

    def forward(self, generated, target):
        # Extract edges from both generated and target images
        edges_generated = self.sobel_filter(generated)
        edges_target = self.sobel_filter(target)

        # Compute the edge loss (L1 loss between edge maps)
        loss = self.criterion(edges_generated, edges_target)
        return loss


def gradient_loss(pred, target):
    # Define Sobel filters
    sobel_x = torch.tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
    ).unsqueeze(0).to(pred.device)  # Add batch and channel dimensions
    sobel_y = torch.tensor(
        [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
    ).unsqueeze(0).to(pred.device)

    # Apply Sobel filters to both predicted and target images
    pred_x = torch.nn.functional.conv2d(pred, sobel_x, padding=1, stride=1)
    pred_y = torch.nn.functional.conv2d(pred, sobel_y, padding=1, stride=1)
    target_x = torch.nn.functional.conv2d(target, sobel_x, padding=1, stride=1)
    target_y = torch.nn.functional.conv2d(target, sobel_y, padding=1, stride=1)

    # Compute L1 loss between gradients
    loss_x = torch.nn.functional.l1_loss(pred_x, target_x)
    loss_y = torch.nn.functional.l1_loss(pred_y, target_y)
    return loss_x + loss_y

def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    return torch.nn.functional.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

def downsample(input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0,
               weight_sd=0.02):
    layers = [nn.Conv2d(input_channels, output_channels, kernel_size, stride=2, padding=1, bias=False)]

    # Initialize the weights with mean and standard deviation
    nn.init.normal_(layers[0].weight, mean=weight_mean, std=weight_sd)

    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.LeakyReLU(0.2))

    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))

    return nn.Sequential(*layers)


def upsample(input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0,
             weight_sd=0.02):
    layers = [nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=2, padding=1, bias=False)]

    # Initialize the weights with mean and standard deviation
    nn.init.normal_(layers[0].weight, mean=weight_mean, std=weight_sd)

    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.ReLU())

    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the downsample layers
        self.conv_layers = nn.ModuleList([
            downsample(1, 64, 4),
            downsample(64, 128, 4),
            downsample(128, 256, 4),
            downsample(256, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4)
        ])

        # Define the upsample layers
        self.up_layers = nn.ModuleList([
            upsample(512, 512, 4),
            upsample(1024, 512, 4),
            upsample(1024, 512, 4),
            upsample(1024, 512, 4),
            upsample(1024, 256, 4),
            # upsample(512, 256, 4),
            upsample(512, 128, 4),
            upsample(256, 64, 4)
        ])

        # Final convolutional layer for generating the output
        self.last = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1)

    def forward(self, x):
        # Downsampling through the model
        skips = []
        for layer in self.conv_layers:
            x = layer(x)
            skips.append(x)

        skips = skips[:-1]

        # Upsampling and establishing skip connections
        for layer, skip in zip(self.up_layers, reversed(skips)):
            x = layer(x)
            x = torch.cat([x, skip], dim=1)

        x = self.last(x)
        return x


##TRAIN
input_files_list = []
output_files_list = []

# Navigate through each subfolder in the master folder
for subfolder in list(sorted(os.listdir(master_folder))):
    subfolder_path = os.path.join(master_folder, subfolder)

    if os.path.isdir(subfolder_path):
        # List all input files with the naming pattern in this subfolder
        input_files = list(sorted([f for f in os.listdir(subfolder_path) if
                                   "Inj_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))
        output_files = list(sorted([f for f in os.listdir(subfolder_path) if
                                    "UV_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))

        for i in range(len(input_files)):
            input_files_list.append(os.path.join(subfolder_path, input_files[i]))
            output_files_list.append(os.path.join(subfolder_path, output_files[i]))

print(f"Number of input files: {len(input_files_list)},", f"Number of output files: {len(output_files_list)}")

input_data = np.array([pd.read_csv(filename, header=None) for filename in input_files_list])
output_data = np.array([pd.read_csv(filename, header=None) for filename in output_files_list])

# unsqueeze the data
input_data = np.expand_dims(input_data, axis=1)
output_data = np.expand_dims(output_data, axis=1)

print(input_data.shape, output_data.shape)

# make dataloader
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# convert to tensor
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# create dataloader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Instantiate the Generator model
# generator = Generator()
generator = VDSR()

# Define the loss function
criterion = nn.L1Loss()
# Initialize edge loss
# edge_loss_fn = EdgeLoss()
# edge_loss_fn = LaplacianPyramidLoss()
# edge_loss_fn = frequency_loss()

# Define the optimizer
optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)

losses = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Move the model to the appropriate device
generator.to(device)

# Define the number of epochs
num_epochs = 300

mse_loss = torch.nn.MSELoss()

# make training loop ( with data loader)
best_test_mse = float('inf')
iter = 0
for epoch in range(num_epochs):
    # running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        output = generator(data)

        # Calculate loss
        L1loss = criterion(output, target)

        # Compute edge loss
        # eloss1 = edge_loss_fn(output, target)
        # eloss = 1 - pytorch_msssim.ms_ssim(output, target, data_range=1.0)
        eloss = gradient_loss(output, target)

        loss = 0.7 * eloss + 0.3 * L1loss# + 0.8 * eloss1

        # loss = L1loss
        # running_loss += loss.item()
        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Log losses
        losses.append(loss.item())

        if iter % 500 == 0:

            print(f"Epoch: {epoch}, loss: {loss.item()}")

            generator.eval()  # Set the generator to evaluation mode
            with torch.no_grad():  # No gradients needed for evaluation
                test_mse = 0.0
                test_samples = 0

                for input_image, target in test_loader:
                    # Transfer to GPU
                    input_image = input_image.to(device)
                    target = target.to(device)

                    # Generate the output using the generator for the current batch
                    gen_output = generator(input_image)

                    # Calculate and accumulate MSE loss
                    # test_mse += mse_loss(gen_output, target).item() * target.size(0)
                    L1loss = criterion(gen_output, target)
                    # eloss1 = edge_loss_fn(gen_output, target)
                    # eloss = 1 - pytorch_msssim.ms_ssim(gen_output, target, data_range=1.0)
                    eloss = gradient_loss(gen_output, target)

                    loss = 0.3 * L1loss + 0.7 * eloss# + 0.8 * eloss1
                    # loss = L1loss
                    test_mse += loss.item() * target.size(0)
                    test_samples += target.size(0)

                # Compute the average MSE over the test set
                test_mse /= test_samples

            print(f'Epoch {epoch}: Average MSE on Test Set: {test_mse}')
            if test_mse < best_test_mse:
                # File path with current date and time
                path = f'pytorch_UNET_ours_ours.pth'
                # Save the model
                torch.save(generator.state_dict(), path)
                best_test_mse = test_mse
                print('Model_Saved')

            generator.train()  # Set the generator back to training mode

        iter += 1

#
# # Current date and time
# current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#
# # File path with current date and time
# path = f'pytorch_UNET.pth'
#
# # Save the model
# torch.save(generator.state_dict(), path)
# #
# # # print(f"Model saved to {path}")


##EVAL


def calculate_contrast(image):
    # Replace this with your chosen method of calculating contrast
    # Using np.nanstd to ignore NaN values
    return np.nanstd(image)

def calculate_mape(y_true, y_pred):
    # Avoid division by zero
    non_zero_mask = y_true != 0
    return torch.mean((torch.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100).item()

#

def generate_images( test_input, tar, prediction, idx):
  plt.figure(figsize=(15, 15))
  display_list = [test_input, tar, prediction]
  global_min = np.min(tar)
  global_max = np.max(tar)
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    if i == 0 or i == 1:
        plt.imshow(display_list[i], cmap='jet', vmin=global_min, vmax=global_max)
    else:
        plt.imshow(display_list[i], cmap='jet', vmin=global_min, vmax=global_max)
    plt.axis('off')
  plt.savefig(f'example_{idx}.png',format='png',dpi=300,bbox_inches='tight')
  plt.close()


def getboxplot(input_energy, output_energy, msemetric, mapemetric, contrast):
    # Create figure and subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # First boxplot for input/output energy
    axes[0].boxplot([np.abs(np.array(output_energy['UNet_L1_E']) - np.array(input_energy))],
                    labels=['ViT-3D Image Denoiser'], showfliers=False)
    axes[0].set_title('Energy Diff. (Joules)', fontsize=15)
    axes[0].set_ylim(0, 15)
    for label in axes[0].get_xticklabels():
        label.set_fontsize(15)
    for label in axes[0].get_yticklabels():
        label.set_fontsize(15)

    # Second boxplot for MSE
    axes[1].boxplot([msemetric['UNet_L1_E']], labels=['ViT-3D Image Denoiser'], showfliers=False)
    axes[1].set_title('MSE', fontsize=15)
    axes[1].set_ylim(0, 0.01)
    for label in axes[1].get_xticklabels():
        label.set_fontsize(15)
    for label in axes[1].get_yticklabels():
        label.set_fontsize(15)

    # Third boxplot for MAPE
    axes[2].boxplot([mapemetric['UNet_L1_E']], labels=['ViT-3D Image Denoiser'], showfliers=False)
    axes[2].set_title('MAPE', fontsize=15)
    axes[2].set_ylim(0, 30)
    for label in axes[2].get_xticklabels():
        label.set_fontsize(15)
    for label in axes[2].get_yticklabels():
        label.set_fontsize(15)

    # Fourth boxplot for contrast
    axes[3].boxplot([contrast['UNet_L1_E']], labels=['ViT-3D Image Denoiser'], showfliers=False)
    axes[3].set_title('Contrast', fontsize=15)
    axes[3].set_ylim(0, 1.3)
    for label in axes[3].get_xticklabels():
        label.set_fontsize(15)
    for label in axes[3].get_yticklabels():
        label.set_fontsize(15)

    # Adjust spacing between plots
    plt.tight_layout()
    plt.savefig('resbox_ours_ours.png', dpi=300, format='png', bbox_inches='tight')
    # Show the plot


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_state_unetL1 = Generator()  # Assuming Generator is a defined class
# model_state_unetL1.load_state_dict(torch.load('pytorch_UNET_correct.pth'))
# model_state_unetL1.eval()
# model_state_unetL1.to(device)

model_state_unetL1E = VDSR()  # Assuming Generator is a defined class
model_state_unetL1E.load_state_dict(torch.load('pytorch_UNET_ours_ours.pth'))
model_state_unetL1E.eval()
model_state_unetL1E.to(device)
models = {
    'UNet_L1_E': model_state_unetL1E
}
mse_loss = torch.nn.MSELoss()
joules_scalar = 0.0333958286584664  # convert energy to joules

master_folder = '../IntIntSpatialTestData'
# master_folder_orig = '../SpatialTestData'
input_files_list = []
output_files_list = []

# Navigate through each subfolder in the master folder
for subfolder in list(sorted(os.listdir(master_folder))):
    subfolder_path = os.path.join(master_folder, subfolder)

    if os.path.isdir(subfolder_path):
        # List all input files with the naming pattern in this subfolder
        input_files = list(sorted([f for f in os.listdir(subfolder_path) if
                                   "Inj_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))
        output_files = list(sorted([f for f in os.listdir(subfolder_path) if
                                    "UV_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))

        for i in range(len(input_files)):
            input_files_list.append(os.path.join(subfolder_path, input_files[i]))
            output_files_list.append(os.path.join(subfolder_path, output_files[i]))

print(f"Number of input files: {len(input_files_list)},", f"Number of output files: {len(output_files_list)}")

input_data = np.array([pd.read_csv(filename, header=None) for filename in input_files_list])
output_data = np.array([pd.read_csv(filename, header=None) for filename in output_files_list])

# unsqueeze the data
input_data = np.expand_dims(input_data, axis=1)
output_data = np.expand_dims(output_data, axis=1)

print(input_data.shape, output_data.shape)

# # make dataloader
# # split data into train and test
# X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# convert to tensor
X_test = torch.from_numpy(input_data).float()
y_test = torch.from_numpy(output_data).float()

# create dataloader
# train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 1
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

input_energy = []
msemetric = defaultdict(list)
mapemetric = defaultdict(list)
output_energy = defaultdict(list)
contrast = defaultdict(list)
all_mapes_distributions = defaultdict(list)
for idx, (test_input_tensor, tar_tensor) in enumerate(test_loader):
    # test_input_tensor = torch.from_numpy(input_data[i]).float().unsqueeze(0).unsqueeze(0).to(device)  # Add batch dimension
    # tar_tensor = torch.from_numpy(output_data[i]).float().unsqueeze(0)
    # images = [test_input_tensor, tar_tensor]
    input_energy.append(np.sum(tar_tensor.cpu().squeeze().numpy()) * joules_scalar)
    for model_name, model in models.items():
        with torch.no_grad():
            prediction_tensor = model(
                test_input_tensor.to(device)).cpu().squeeze()  # Generate prediction and move to CPU
            mse = mse_loss(prediction_tensor, tar_tensor.cpu().squeeze()).item()
            mape = calculate_mape(tar_tensor.cpu().squeeze(), prediction_tensor)
            msemetric[model_name].append(mse)
            mapemetric[model_name].append(mape)
            output_energy[model_name].append(np.sum(prediction_tensor.numpy()) * joules_scalar)
            contrast[model_name].append(calculate_contrast(prediction_tensor.numpy()))
            # generate_images(test_input_tensor.cpu().squeeze(0).squeeze(0).numpy(),
            #                 tar_tensor.cpu().squeeze(0).squeeze(0).numpy(), prediction_tensor.cpu().numpy(), idx)
            # all_mapes_distributions[model_name].append(mape_distribution(prediction_tensor.cpu().numpy(), tar_tensor.cpu().squeeze(0).numpy()))

getboxplot(input_energy, output_energy, msemetric, mapemetric, contrast)
# # show_mape_dis(all_mapes_distributions)
# #
#
# # path_selections = [code_to_path(code) for code in selections]
# # for i in range(len(path_selections)):
# #
# #     input_path, output_path = path_selections[i]
# #     input_data = load_data_from_one_path(input_path)
# #     output_data = load_data_from_one_path(output_path)
# #
# #     test_input_tensor = input_data.unsqueeze(0).unsqueeze(0).to(device)  # Add batch dimension
# #     tar_tensor = output_data.unsqueeze(0)
# #
# #     # test_input = test_input_tensor.cpu().squeeze().numpy()  # Move input to CPU
# #     # tar = tar_tensor.cpu().squeeze()   # Move target to CPU
# #
# #     images = [test_input_tensor, tar_tensor]
# #     titles = ['Input Image', 'Ground Truth']
# #
# #     print(f'Image index: {selections[i]}')
# #     print(
# #         f'Output pixel intensity of the ground truth (sum of the pixels): {np.sum(tar_tensor.cpu().squeeze().numpy())}')
# #     print(f'Output energy of the ground truth (joules): {np.sum(tar_tensor.cpu().squeeze().numpy()) * joules_scalar}')
# #
# #     for model_name, model in models.items():
# #         with torch.no_grad():
# #             prediction_tensor = model(test_input_tensor).cpu().squeeze()  # Generate prediction and move to CPU
# #             mse = mse_loss(prediction_tensor, tar_tensor.cpu().squeeze()).item()
# #             mape = calculate_mape(tar_tensor.cpu().squeeze(), prediction_tensor)
# #
# #             print(
# #                 f'Pixel intensity of {model_name} prediction (sum of the pixels): {np.sum(prediction_tensor.numpy())}')
# #             print(f'Energy of {model_name} prediction (joules): {np.sum(prediction_tensor.numpy()) * joules_scalar}')
# #             print(f'{model_name} MSE: {mse}, MAPE: {mape}%, Contrast: {calculate_contrast(prediction_tensor.numpy())}')
# #
# #             # prediction = prediction_tensor.numpy()
# #             images.append(prediction_tensor)
# #             titles.append(f'{model_name} Predicted Image')
# #
# #     images = [tensor.cpu().squeeze().numpy() for tensor in images]
# #
# #     # Use the functions
# #     generate_images_samescale(images, titles, selections[i])  # Same color scale for all images (except input)
# #     #add_histograms(images, titles)  # Histograms for each image









# Dataset creation
# import shutil
# def process_and_save_csv(main_folder, csv_file_path, out_file_path):
#     # Extract file name from the path and parse the required parts
#     file_name = os.path.basename(csv_file_path)
#     base_name = file_name.split('_')
#
#     # Extract "Run_20" and convert to desired folder name format
#     run_part = f"{base_name[0].replace('_', ' ')} {base_name[1]} - {base_name[2]}, {base_name[3]}"
#
#     # Create the full path to the directory where the file will be saved
#     target_directory = os.path.join(main_folder, run_part)
#
#     # Check if the directory exists, if not, create it
#     if not os.path.exists(target_directory):
#         os.makedirs(target_directory)
#
#     # Save the CSV file in the target directory
#     target_file_path = os.path.join(target_directory, file_name)
#     shutil.copy(out_file_path, target_directory)
#     if os.path.exists(target_file_path):
#         print('error')
#         return -1
#
#     return target_file_path
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# model_state_unetL1E = Generator()  # Assuming Generator is a defined class
# model_state_unetL1E.load_state_dict(torch.load('pytorch_UNET_ours.pth'))
# model_state_unetL1E.eval()
# model_state_unetL1E.to(device)
# models = {
#     'UNet_L1_E': model_state_unetL1E
# }
# mse_loss = torch.nn.MSELoss()
# joules_scalar = 0.0333958286584664  # convert energy to joules
#
# master_folder = '../IntSpatialData'
# tar_folder = '../IntIntSpatialData'
# input_files_list = []
# output_files_list = []
#
# # Navigate through each subfolder in the master folder
# for subfolder in list(sorted(os.listdir(master_folder))):
#     subfolder_path = os.path.join(master_folder, subfolder)
#
#     if os.path.isdir(subfolder_path):
#         # List all input files with the naming pattern in this subfolder
#         input_files = list(sorted([f for f in os.listdir(subfolder_path) if
#                                    "Inj_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))
#         output_files = list(sorted([f for f in os.listdir(subfolder_path) if
#                                     "UV_256x256_InjEnergyFactor_" in f and f.endswith(".csv")]))
#
#         for i in range(len(input_files)):
#             input_files_list.append(os.path.join(subfolder_path, input_files[i]))
#             output_files_list.append(os.path.join(subfolder_path, output_files[i]))
#
# print(f"Number of input files: {len(input_files_list)},", f"Number of output files: {len(output_files_list)}")
#
# for inp,out in zip(input_files_list, output_files_list):
#     input_data = np.array([pd.read_csv(inp, header=None)])
#     input_data = np.expand_dims(input_data, axis=1)
#     input_tensor = torch.from_numpy(input_data).float()
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         prediction_tensor = model_state_unetL1E(input_tensor).cpu().squeeze()  # Generate prediction and move to CPU
#     df = pd.DataFrame(prediction_tensor.numpy())
#     save_name = process_and_save_csv(tar_folder, inp, out)
#     df.to_csv(save_name, index=False, header=False)

