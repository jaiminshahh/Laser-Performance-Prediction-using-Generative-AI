import torch.nn as nn
from models import UNet3D, ViT3D#, ProposedVnet
from torch import optim
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from metric import calculate_mape, calculate_contrast
from loss import LaplacianPyramidLoss, SSIMLoss, EdgeLoss
import matplotlib.pyplot as plt
import time


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader, device):
        # Data loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # Model name
        self.model_type = config['model']['name']
        # Device
        self.device = device
        # Losses. Can be imported from loss.py
        self.criterion = nn.L1Loss()
        # self.criterion = nn.MSELoss()
        # self.smoothness = LaplacianPyramidLoss()
        # self.smoothness = SSIMLoss()
        self.smoothness = EdgeLoss()
        # Training params settings
        self.lr = config['training_parameters']['learning_rate']
        self.num_epochs = config['training_parameters']['epochs']
        self.batch_size = config['training_parameters']['batch_size']
        # Model initialization
        self.build_model()


    def build_model(self):
        # List of different networks we use. Imported from models.py
        if self.model_type == 'UNet-3D':
            self.net = UNet3D(in_channels=1, out_channels=1, base_filters=8)
        elif self.model_type == 'ViT-3D':
            self.net = ViT3D()
        # elif self.model_type == "ProposedVNet":
        #     self.net = ProposedVnet(image_size=256, slice_depth_size=600,
        #                             image_patch_size=16,
        #                             slice_depth_patch_size=50, dim=768, depth=1, heads=2,
        #                             mlp_dim=768, channels=1, dim_head=64)
        else:
            print('Model Not Known')
            return -1
        # Set the optimizer and load the model to cuda
        self.optimizer = optim.Adam(self.net.parameters(), self.lr)
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
        self.net.to(self.device)


    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.net.train()
            total_loss = 0; val_loss = 0
            for inp_vid, tar_img, tar_pls, _ in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                inp_vid, tar_img, tar_pls = inp_vid.to(self.device), tar_img.to(self.device), tar_pls.to(self.device)
                # Takes input video and gives output video
                out_vid = self.net(inp_vid)
                # Spatial and temporal averaging
                out_img = out_vid.mean(dim=2, keepdim=True).squeeze(1).squeeze(1)
                # Norm factor scaling
                out_pls = out_vid.mean(dim=[3,4], keepdim=True).squeeze(1).squeeze(2).squeeze(2)
                # norm_factor = torch.tensor([(torch.sum(x) * 0.03) / (torch.sum(y) * 0.025) for x, y in zip(tar_img, tar_pls)])
                # norm_factor = torch.tensor([(torch.sum(x)) / (torch.sum(y)) for x, y in zip(tar_pls, out_pls)])
                # norm_factor = torch.tensor([0.8983,0.8983,0.8983,0.8983])
                # out_pls = out_pls * norm_factor.unsqueeze(1).to(self.device)
                # Compute the spatial and temporal loss
                loss_img = self.criterion(out_img, tar_img)
                loss_pls = self.criterion(out_pls, tar_pls)
                # Smoothness loss for image
                # loss_smooth = self.smoothness(out_img, tar_img)
                # Total loss. Can play with factors
                loss = 0.5*loss_img + 0.5*loss_pls
                # loss = loss_pls
                # loss = loss_img
                total_loss += loss.item()
                # Optimization
                loss.backward()
                self.optimizer.step()
            print(f"Epoch: {epoch + 1}/ {self.num_epochs}, Loss: {total_loss/len(self.train_loader)}")

            # Validation after each epoch. Same as training. Save the model with least validation loss
            self.net.eval()
            with torch.no_grad():
                for inp_vid, tar_img, tar_pls, _ in tqdm(self.valid_loader):
                    inp_vid, tar_img, tar_pls = inp_vid.to(self.device), tar_img.to(self.device), tar_pls.to(self.device)
                    out_vid = self.net(inp_vid)
                    out_img = out_vid.mean(dim=2, keepdim=True).squeeze(1).squeeze(1)
                    # out_pls = out_vid.mean(dim=[3, 4], keepdim=True).squeeze(1).squeeze(2).squeeze(2)
                    loss_img = self.criterion(out_img, tar_img)
                    # norm_factor = torch.tensor([(torch.sum(x) * 0.03) / (torch.sum(y) * 0.025) for x, y in zip(tar_img, tar_pls)])
                    # norm_factor = torch.tensor([(torch.sum(x)) / (torch.sum(y)) for x, y in zip(tar_pls, out_pls)])
                    # norm_factor = torch.tensor([0.8983, 0.8983, 0.8983, 0.8983])
                    # out_pls = out_pls * norm_factor.unsqueeze(1).to(self.device)
                    loss_pls = self.criterion(out_pls, tar_pls)
                    # loss_smooth = self.smoothness(out_img, tar_img)
                    loss = 0.5*loss_img + 0.5*loss_pls
                    # loss = loss_pls
                    # loss = loss_img
                    val_loss += loss.item()
                val_loss = val_loss/len(self.valid_loader)
                print(f"Epoch: {epoch + 1} Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.net.state_dict(), f"model_{self.model_type}.pth")
                    print(f"Model Saved")


    def test(self):
        self.net.load_state_dict(torch.load(f"model_{self.model_type}.pth"))
        self.net.eval()
        input_energy_image = []; input_energy_pulse = []
        msemetric_image=[]; msemetric_pls = []; mapemetric=[]
        output_energy_image=[]; contrast=[]
        timetaken = []; output_energy_pulse = []
        with torch.no_grad():
            for inp_vid, tar_img, tar_pls, inp_pls in tqdm(self.test_loader):
                inp_vid = inp_vid.to(self.device); tar_img = tar_img.to(self.device); tar_pls = tar_pls.to(self.device); inp_pls = inp_pls.to(self.device)
                input_energy_image.append(np.sum(tar_img.cpu().squeeze().numpy()) * 0.03)
                input_energy_pulse.append(np.sum(tar_pls.cpu().numpy()) * 0.025)
                start_time = time.time()
                out_vid = self.net(inp_vid)  # Generate prediction and move to CPU
                out_img = out_vid.sum(dim=2, keepdim=True).squeeze(1).squeeze(1)
                out_pls = out_vid.sum(dim=[3, 4], keepdim=True).squeeze(1).squeeze(2).squeeze(2)
                end_time = time.time()
                timetaken.append((end_time - start_time)*1000)
                norm_factor = torch.tensor([(torch.sum(x) * 0.03) / (torch.sum(y) * 0.025) for x, y in zip(out_img, out_pls)])
                out_pls = out_pls * norm_factor.unsqueeze(1).to(self.device)
                # out_pls = self.smooth_pulse(out_pls)
                msemetric_image.append(nn.MSELoss()(out_img.squeeze(0), tar_img.squeeze(0)).item())
                mapemetric.append(calculate_mape(tar_img.squeeze(0).cpu(), out_img.squeeze(0).cpu()))
                output_energy_image.append(np.sum(out_img.cpu().squeeze().numpy()) * 0.03)
                msemetric_pls.append(nn.MSELoss()(out_pls.squeeze(0), tar_pls.squeeze(0)).item())
                output_energy_pulse.append(np.sum(out_pls.cpu().numpy()) * 0.025)
                contrast.append(calculate_contrast(out_img.squeeze(0).cpu().numpy()))
                self.plotimgpls(inp_vid, tar_img, out_img, tar_pls, out_pls, inp_pls)
            self.getboxplot_image(input_energy_image, output_energy_image, msemetric_image, mapemetric, contrast, timetaken)
            self.getboxplot_pls(input_energy_pulse, output_energy_pulse, msemetric_pls)

    def smooth_pulse(self, pulse, kernel_size=5, sigma=1.0):
        # Create a 1D Gaussian kernel
        x = torch.arange(kernel_size) - kernel_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_kernel = (gauss/gauss.sum()).unsqueeze(0).unsqueeze(0).to(self.device)
        # Apply the Gaussian smoothing using conv1d
        smoothed_pulse = F.conv1d(pulse.unsqueeze(1), gauss_kernel, padding=kernel_size // 2)

        return smoothed_pulse.squeeze(1)

    def plotimgpls(self, inp_vid, tar_img, out_img, tar_pls, out_pls, inp_pls):
        plt.figure(figsize=(15, 15))
        display_list = [inp_vid[:, :, 300, :, :].squeeze(0).squeeze(0).cpu().numpy(), tar_img.cpu().squeeze(0).numpy(),
                        out_img.cpu().squeeze(0).numpy()]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='jet')
            # plt.axis('off')
        plt.savefig('img.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(20, 10))
        display_list = [inp_pls.cpu().squeeze(0).numpy(),
            tar_pls.cpu().squeeze(0).numpy(),
            out_pls.cpu().squeeze(0).numpy()]
        title = ['Input Pulse', 'Ground Truth', 'Predicted Pulse']
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.plot(range(576), display_list[i])
            # plt.axis('off')
            plt.ylabel('Power')
        plt.savefig('pls.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def getboxplot_image(self, input_energy, output_energy, msemetric, mapemetric, contrast, timetaken):
        # Create figure and subplots
        fig, axes = plt.subplots(1, 5, figsize=(16, 5))

        # First boxplot for input/output energy
        axes[0].boxplot([(np.abs(np.array(output_energy) - np.array(input_energy))/np.array(input_energy))*100], labels=['ViT-3D'], showfliers=True)
        axes[0].set_ylim(0, 20)
        axes[0].set_title('Input/Output Energy Difference (%)')

        # Second boxplot for MSE
        axes[1].boxplot(msemetric, labels=['ViT-3D'], showfliers=True)
        axes[1].set_ylim(0, 0.01)
        axes[1].set_title('MSE')

        # Third boxplot for MAPE
        axes[2].boxplot(mapemetric, labels=['ViT-3D'], showfliers=True)
        axes[2].set_ylim(0,60)
        axes[2].set_title('MAPE (%)')

        # Fourth boxplot for contrast
        axes[3].boxplot(contrast, labels=['ViT-3D'], showfliers=True)
        axes[3].set_ylim(0, 1.0)
        axes[3].set_title('Contrast')

        # Fifth boxplot for inference time
        axes[4].boxplot(timetaken, labels=['ViT-3D'], showfliers=True)
        axes[4].set_ylim(0, 50)
        axes[4].set_title('Inference Time (ms)')

        # Adjust spacing between plots
        plt.tight_layout()
        plt.savefig('resbox_image.png', dpi=300, format='png', bbox_inches='tight')
        # Show the plot


    def getboxplot_pls(self, input_energy, output_energy, msemetric):
        # Create figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(5, 5))

        # First boxplot for input/output energy
        axes[0].boxplot([(np.abs(np.array(output_energy) - np.array(input_energy))/np.array(input_energy))*100], labels=['ViT-3D'], showfliers=True)
        axes[0].set_ylim(0, 20)
        axes[0].set_title('Input/Output Energy Difference (%)')

        # Second boxplot for MSE
        axes[1].boxplot(msemetric, labels=['ViT-3D'], showfliers=True)
        axes[1].set_ylim(0, 0.01)
        axes[1].set_title('MSE')

        # Adjust spacing between plots
        plt.tight_layout()
        plt.savefig('resbox_pls.png', dpi=300, format='png', bbox_inches='tight')
        # Show the plot