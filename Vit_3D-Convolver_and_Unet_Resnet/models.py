import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
# from timm.models.swin_transformer import SwinTransformer3D

class ViT3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(64, 4, 4), emb_dim=128, depth=2, n_heads=2, mlp_dim=512):
        super(ViT3D, self).__init__()

        # Input parameters
        self.in_channels = in_channels
        self.preconv_channels = 32
        self.patch_size = patch_size  # (depth, height, width)
        self.emb_dim = emb_dim
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.inp_depth  = 576
        self.inp_height = 256
        self.inp_width = 256
        # self.stride  =(50,4,4)

        # Pre-transformer conv
        self.pre_transformer_conv = nn.Conv3d(in_channels, self.preconv_channels, kernel_size=3, padding=1)

        # Compute the number of patches in each dimension
        self.num_patches_d = (self.inp_depth + patch_size[0] - 1) // patch_size[0]
        self.num_patches_h = (self.inp_height + patch_size[1] - 1) // patch_size[1]
        self.num_patches_w = (self.inp_width + patch_size[2] - 1) // patch_size[2]
        self.n_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w
        # self.num_patches_d = ((self.inp_depth - patch_size[0]) // self.stride[0]) + 1
        # self.num_patches_h = ((self.inp_height - patch_size[1]) // self.stride[1]) + 1
        # self.num_patches_w = ((self.inp_width - patch_size[2]) // self.stride[2]) + 1
        # self.n_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w

        # Patch embedding layer
        self.patch_embedding = nn.Conv3d(
            self.preconv_channels, emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Regression head for per-patch predictions
        self.regression_head = nn.Linear(emb_dim, patch_size[0] * patch_size[1] * patch_size[2] * self.preconv_channels)

        # Post transformer conv
        self.post_transformer_conv = nn.Conv3d(self.preconv_channels, 1, kernel_size=3, padding=1)


    def forward(self, x):
        # x shape: (batch_size, in_channels, depth, height, width)
        batch_size, _, D, H, W = x.size()

        # Preconv
        x = self.pre_transformer_conv(x)

        # Calculate required padding for each dimension
        # pad_d = (self.patch_size[0] - D % self.patch_size[0]) % self.patch_size[0]
        # pad_h = (self.patch_size[1] - H % self.patch_size[1]) % self.patch_size[1]
        # pad_w = (self.patch_size[2] - W % self.patch_size[2]) % self.patch_size[2]

        # Apply padding
        # x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        # Patch embedding
        x = self.patch_embedding(x)  # Shape: (batch_size, emb_dim, D_patches, H_patches, W_patches)

        # Flatten patches
        x = x.flatten(2)  # Shape: (batch_size, emb_dim, n_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, n_patches, emb_dim)

        # Add positional embeddings
        x = x + self.pos_embedding  # Shape: (batch_size, n_patches, emb_dim)

        # Transformer encoding
        x = self.transformer(x)  # Shape: (batch_size, n_patches, emb_dim)

        # Regression for each patch
        x = self.regression_head(x)  # Shape: (batch_size, n_patches, patch_volume)

        # Reshape to reconstruct the volume
        x = x.view(batch_size, self.num_patches_d, self.num_patches_h, self.num_patches_w, self.patch_size[0], self.patch_size[1], self.patch_size[2], self.preconv_channels)
        # Shape: (batch_size, D_patches, H_patches, W_patches, pd, ph, pw)

        # Rearrange dimensions to match the original volume structure
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (batch_size, D_patches, pd, H_patches, ph, W_patches, pw)
        x = x.contiguous().view(batch_size, self.preconv_channels, self.num_patches_d * self.patch_size[0], self.num_patches_h * self.patch_size[1], self.num_patches_w * self.patch_size[2])
        # Shape: (batch_size, 1, D_out, H_out, W_out)

        # Remove padding to match original input dimensions
        # x = x[:, :, int((x.shape[2] - D)/2):int((x.shape[2] - D)/2)+D
        #           , int((x.shape[3] - H)/2):int((x.shape[3] - H)/2)+H
        #           , int((x.shape[4] - W)/2):int((x.shape[4] - W)/2)+W ]
        # Post conv
        x = self.post_transformer_conv(x)

        return x



class Discriminator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Discriminator, self).__init__()

        # The first few convolutional layers using the downsample block
        self.downsample1 = self.downsample(input_channels * 2, 64, 4, apply_batchnorm=False)
        self.downsample2 = self.downsample(64, 128, 4)
        self.downsample3 = self.downsample(128, 256, 4)

        # Continue with the remaining convolutional layers
        self.conv4 = nn.Conv2d(256, 512, 4, stride=1, padding=0)  # No padding here
        self.conv5 = nn.Conv2d(512, 1, 4, stride=1, padding=0)

        # Batch normalization layer
        self.batchnorm1 = nn.BatchNorm2d(512)

        # Activation function
        self.leaky_relu = nn.LeakyReLU(0.2)

    def downsample(self, input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0,
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

    def forward(self, input_image, target_image):
        input_image = input_image.unsqueeze(1)
        target_image = target_image.unsqueeze(1)
        # Concatenate input and target images along the channel dimension
        x = torch.cat((input_image, target_image), dim=1)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = nn.ZeroPad2d((1, 1, 1, 1))(x)

        # Continue with the remaining convolutional layers
        x = self.conv4(x)
        x = self.batchnorm1(x)
        x = self.leaky_relu(x)
        x = nn.ZeroPad2d((1, 1, 1, 1))(x)
        x = self.conv5(x)

        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, base_filters)  # Layer 1
        self.encoder2 = self.conv_block(base_filters, base_filters * 2)  # Layer 2
        self.encoder3 = self.conv_block(base_filters * 2, base_filters * 4)  # Layer 3
        self.encoder4 = self.conv_block(base_filters * 4, base_filters * 8)  # Layer 4
        self.encoder5 = self.conv_block(base_filters * 8, base_filters * 8)  # Layer 5
        self.encoder6 = self.conv_block(base_filters * 8, base_filters * 8)  # Layer 6
        self.encoder7 = self.conv_block(base_filters * 8, base_filters * 8)  # Layer 7
        self.encoder8 = self.conv_block(base_filters * 8, base_filters * 8)  # Layer 8
        # Bottleneck
        # self.bottleneck = self.conv_block(base_filters * 8, base_filters * 16)

        # Decoder
        self.upconv7 = self.upconv_block(base_filters * 8, base_filters * 8) # Layer 7

        self.upconv6 = self.upconv_block(base_filters * 16, base_filters * 8) # Layer 6

        self.upconv5 = self.upconv_block(base_filters * 16, base_filters * 8) # Layer 5

        self.upconv4 = self.upconv_block(base_filters * 16, base_filters * 8)  # Layer 4
        # self.decoder4 = self.conv_block(base_filters * 16, base_filters * 8)

        self.upconv3 = self.upconv_block(base_filters * 16, base_filters * 4)  # Layer 3
        # self.decoder3 = self.conv_block(base_filters * 8, base_filters * 4)

        self.upconv2 = self.upconv_block(base_filters * 8, base_filters * 2)  # Layer 2
        # self.decoder2 = self.conv_block(base_filters * 4, base_filters * 2)

        self.upconv1 = self.upconv_block(base_filters * 4, base_filters)  # Layer 1
        # self.decoder1 = self.conv_block(base_filters * 2, base_filters)

        # Final output layer
        self.final_conv = nn.ConvTranspose3d(base_filters * 2, out_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))

    def conv_block(self, in_channels, out_channels):
        """
        Convolution block with Conv3D, BatchNorm3D, and ReLU.
        Kernel size: (50, 4, 4)
        """
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """
        Transposed convolution (upsampling) layer.
        """
        return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),
                             nn.BatchNorm3d(out_channels),
                             nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        enc8 = self.encoder8(enc7)

        # Bottleneck
        # bottleneck = self.bottleneck(enc4)

        # Decoder path (upsampling + concatenation)
        dec7 = self.upconv7(enc8)
        dec7 = torch.cat((dec7, enc7), dim=1)

        dec6 = self.upconv6(dec7)
        dec6 = torch.cat((torch.cat((dec6, dec6[:,:,7:8,:,:]), dim=2), enc6), dim=1)

        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)

        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        # dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        # dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        # dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.decoder1(dec1)

        # Final output
        out = self.final_conv(dec1)
        return out


# def pair(t):
#     return t if isinstance(t, tuple) else (t, t)
#
# def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
#     _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype
#
#     z, y, x = torch.meshgrid(
#         torch.arange(f, device = device),
#         torch.arange(h, device = device),
#         torch.arange(w, device = device),
#     indexing = 'ij')
#
#     fourier_dim = dim // 6
#
#     omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
#     omega = 1. / (temperature ** omega)
#
#     z = z.flatten()[:, None] * omega[None, :]
#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :]
#
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)
#
#     pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
#     return pe.type(dtype)
#
#
# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#     def forward(self, x):
#         return self.net(x)
#
#
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)
#
#         self.attend = nn.Softmax(dim = -1)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)
#
#     def forward(self, x):
#         x = self.norm(x)
#
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim, heads = heads, dim_head = dim_head),
#                 FeedForward(dim, mlp_dim)
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return self.norm(x)
#
#
#
# class SimpleViT(nn.Module):
#     def __init__(self, image_size, image_patch_size, slice_depth_size, slice_depth_patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(image_patch_size)
#
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#         assert slice_depth_size % slice_depth_patch_size == 0, 'Frames must be divisible by the frame patch size'
#
#         num_patches = (image_height // patch_height) * (image_width // patch_width) * (slice_depth_size // slice_depth_patch_size)
#         patch_dim = channels * patch_height * patch_width * slice_depth_patch_size
#
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = slice_depth_patch_size),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
#
#
#
#     def forward(self, video):
#         *_, h, w, dtype = *video.shape, video.dtype
#
#         x = self.to_patch_embedding(video)
#         pe = posemb_sincos_3d(x)
#         x = rearrange(x, 'b ... d -> b (...) d') + pe
#
#         x = self.transformer(x)
#         return x
#
#
#
# class ProposedVnet(nn.Module):
#     def __init__(self, image_size, slice_depth_size, image_patch_size, slice_depth_patch_size, dim, depth, heads,
#                  mlp_dim, channels, dim_head, num_classes, survival_classes):
#         super().__init__()
#         self.image_size=image_size
#         self.slice_depth_size=slice_depth_size
#         self.image_patch_size=image_patch_size
#         self.slice_depth_patch_size=slice_depth_patch_size
#         self.numclasses = num_classes
#
#         self.vit3d = SimpleViT(image_size=image_size, image_patch_size=image_patch_size, slice_depth_size=slice_depth_size,
#                                slice_depth_patch_size=slice_depth_patch_size,
#                                dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head)
#         self.downconv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, stride=2, padding=1),
#                                       nn.BatchNorm3d(dim),
#                                       nn.ReLU(inplace=True))
#         self.upconv = nn.Sequential(nn.ConvTranspose3d(dim, int(dim/2), kernel_size=4, stride=2, padding=1, output_padding=0),
#                                     nn.BatchNorm3d(int(dim/2)),
#                                     nn.ReLU(inplace=True))
#         self.lastconv = nn.Sequential(nn.ConvTranspose3d(dim, num_classes, kernel_size=16, stride=8, padding=4, output_padding=0))
#         self.onedownconv = nn.Sequential(nn.Conv3d(dim, int(dim/2), kernel_size=1, stride=1, padding=0),
#                                          nn.BatchNorm3d(int(dim/2)),
#                                          nn.ReLU(inplace=True))
#         # Regression head for per-patch predictions
#         self.regression_head = nn.Sequential(nn.Linear(dim, image_patch_size * image_patch_size * slice_depth_patch_size),
#                                              nn.ReLU(inplace=True))
#
#
#     def forward(self, x):
#         t1 = self.vit3d(x)
#
#         t1d = rearrange(t1, 'b (d h w) k -> b k d h w', d=int(self.slice_depth_size / self.slice_depth_patch_size),
#                         h=int(self.image_size / self.image_patch_size))
#         t1dc = self.downconv(t1d)
#         t1dc = rearrange(t1dc, 'b k d h w -> b (d h w) k')
#
#         t2 = self.vit3d.transformer(t1dc)
#
#         t2u = rearrange(t2, 'b (d h w) k -> b k d h w',
#                         d=int(self.slice_depth_size / (2 * self.slice_depth_patch_size)),
#                         h=int(self.image_size / (2 * self.image_patch_size)))
#         t2uc = self.upconv(t2u)
#         t2ucat = torch.cat((self.onedownconv(t1d), t2uc), dim=1)
#         t2ucat = rearrange(t2ucat, 'b k d h w -> b (d h w) k')
#
#         t3 = self.vit3d.transformer(t2ucat)
#
#         # t3u = rearrange(t3, 'b (d h w) k -> b k d h w', d=int(self.slice_depth_size / (self.slice_depth_patch_size)),
#         #                 h=int(self.image_size / (self.image_patch_size)))
#         out = self.regression_head(t3)
#         return out


class ViT3DUNet(nn.Module):
    def __init__(self, base_model_path):
        super(ViT3DUNet, self).__init__()
        self.base_net = ViT3D()
        ckpt = torch.load(base_model_path)
        new_state_dict = {}
        for key, value in ckpt.items():
            new_key = key.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = value
        self.base_net.load_state_dict(new_state_dict)
        for param in self.base_net.parameters():
            param.requires_grad = False
        self.generator = Generator()

    def forward(self, x):
        x = self.base_net(x)
        x_img = x.mean(dim=2, keepdim=True).squeeze(1)
        # x_pls = x.mean(dim=[3, 4], keepdim=True).squeeze(1).squeeze(2).squeeze(2)
        out = self.generator(x_img)
        return out.squeeze(1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the downsample layers
        self.conv_layers = nn.ModuleList([
            self.downsample(1, 64, 4),
            self.downsample(64, 128, 4),
            self.downsample(128, 256, 4),
            self.downsample(256, 512, 4),
            self.downsample(512, 512, 4),
            self.downsample(512, 512, 4),
            self.downsample(512, 512, 4),
            self.downsample(512, 512, 4)
        ])

        # Define the upsample layers
        self.up_layers = nn.ModuleList([
            self.upsample(512, 512, 4),
            self.upsample(1024, 512, 4),
            self.upsample(1024, 512, 4),
            self.upsample(1024, 512, 4),
            self.upsample(1024, 256, 4),
            self.upsample(512, 128, 4),
            self.upsample(256, 64, 4)
        ])

        # Final convolutional layer for generating the output
        self.last = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1)

    def downsample(self, input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0,
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

    def upsample(self, input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0,
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