import torch.nn as nn
import torch
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True):
        """
        Initializes the SSIM loss with a Gaussian filter window.
        Args:
            window_size (int): Size of the Gaussian window for SSIM computation.
            sigma (float): Standard deviation for Gaussian window.
            size_average (bool): If True, averages the SSIM loss across the batch.
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 1  # By default for grayscale images (adjustable for multi-channel images)
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        """
        Creates a 2D Gaussian window.
        Args:
            window_size (int): Size of the Gaussian window.
            channel (int): Number of channels in the input image (1 for grayscale, 3 for RGB).
        Returns:
            Tensor: The Gaussian window as a 4D tensor.
        """
        # Create a 1D tensor for the window
        _1D_window = torch.arange(window_size, dtype=torch.float32) - window_size // 2

        # Apply the Gaussian function to create the Gaussian window
        _1D_window = torch.exp(-(_1D_window ** 2) / (2 * self.sigma ** 2))

        # Normalize the window
        _1D_window = _1D_window / _1D_window.sum()

        # Create a 2D Gaussian window
        _2D_window = _1D_window.unsqueeze(1).mm(_1D_window.unsqueeze(0))

        # Convert the 2D window to 4D and expand for the number of channels
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()

        return window

    def ssim(self, pred, target, window, window_size, channel):
        """
        Computes the SSIM between predicted and target images.
        Args:
            pred (Tensor): The predicted image of shape (batch_size, channels, height, width).
            target (Tensor): The target image of shape (batch_size, channels, height, width).
            window (Tensor): The Gaussian window used for SSIM.
            window_size (int): Size of the Gaussian window.
            channel (int): Number of channels in the input image.
        Returns:
            Tensor: SSIM index between predicted and target images.
        """
        # Compute mean values using the Gaussian window
        mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variance and covariance
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, pred, target):
        """
        Computes the SSIM loss between predicted and target images.
        Args:
            pred (Tensor): The predicted image of shape (batch_size, channels, height, width).
            target (Tensor): The target image of shape (batch_size, channels, height, width).
        Returns:
            Tensor: SSIM loss (1 - SSIM index).
        """
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        if pred.size(1) != self.channel:  # Adjust window size if the input is multi-channel
            self.window = self.create_window(self.window_size, pred.size(1)).to(pred.device)
            self.channel = pred.size(1)

        # Compute SSIM index
        ssim_index = self.ssim(pred, target, self.window.to(pred.device), self.window_size, self.channel)

        # SSIM loss is defined as 1 - SSIM
        return 1 - ssim_index


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
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        # Apply Laplacian filter to both predicted and target images
        pred_laplacian = self.laplacian_filter(pred)
        target_laplacian = self.laplacian_filter(target)

        # Compute mean squared error loss between the Laplacian-filtered images
        loss = F.mse_loss(pred_laplacian, target_laplacian)
        return loss

# class LaplacianPyramidLoss(nn.Module):
#     def __int__(self):
#         super(LaplacianPyramidLoss, self).__int__()
#         laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
#         self.register_buffer('laplacian_kernel', laplacian_kernel)
#
#     def laplacian_filter(self, image):
#         laplacian_kernel = self.laplacian_kernel.unsqueeze(0).unsqueeze(0).cuda()  # Shape: (1, 1, 3, 3)
#         image_laplacian = F.conv2d(image, laplacian_kernel, padding=1)
#         return image_laplacian
#
#     # Apply Laplacian loss
#     def laplacian_loss(self, pred, target):
#         return F.mse_loss(self.laplacian_filter(pred), self.laplacian_filter(target))


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
        generated = generated.unsqueeze(1)
        target = target.unsqueeze(1)
        # Extract edges from both generated and target images
        edges_generated = self.sobel_filter(generated)
        edges_target = self.sobel_filter(target)

        # Compute the edge loss (L1 loss between edge maps)
        loss = self.criterion(edges_generated, edges_target)
        return loss