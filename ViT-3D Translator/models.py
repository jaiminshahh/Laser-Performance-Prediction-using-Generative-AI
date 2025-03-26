import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from typing import Tuple

class ViT3D(nn.Module):
    def __init__(self, in_channels=1, patch_size=(50, 16, 16), emb_dim=768, depth=2, n_heads=2, mlp_dim=3072):
        super(ViT3D, self).__init__()

        # Input parameters
        self.in_channels = in_channels
        self.patch_size = patch_size  # (depth, height, width)
        self.emb_dim = emb_dim
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.inp_depth  = 600
        self.inp_height = 256
        self.inp_width = 256

        # Compute the number of patches in each dimension
        self.num_patches_d = (self.inp_depth + patch_size[0] - 1) // patch_size[0]
        self.num_patches_h = (self.inp_height + patch_size[1] - 1) // patch_size[1]
        self.num_patches_w = (self.inp_width + patch_size[2] - 1) // patch_size[2]
        self.n_patches = self.num_patches_d * self.num_patches_h * self.num_patches_w

        # Patch embedding layer
        self.patch_embedding = nn.Conv3d(
            in_channels, emb_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, emb_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Regression head for per-patch predictions
        self.regression_head = nn.Sequential(nn.Linear(emb_dim, patch_size[0] * patch_size[1] * patch_size[2]), nn.ReLU(inplace=True))

    def forward(self, x):
        # x shape: (batch_size, in_channels, depth, height, width)
        batch_size, _, D, H, W = x.size()

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
        x = x.view(batch_size, self.num_patches_d, self.num_patches_h, self.num_patches_w, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        # Shape: (batch_size, D_patches, H_patches, W_patches, pd, ph, pw)

        # Rearrange dimensions to match the original volume structure
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # (batch_size, D_patches, pd, H_patches, ph, W_patches, pw)
        x = x.contiguous().view(batch_size, 1, self.num_patches_d * self.patch_size[0], self.num_patches_h * self.patch_size[1], self.num_patches_w * self.patch_size[2])
        # Shape: (batch_size, 1, D_out, H_out, W_out)

        # Remove padding to match original input dimensions
        # x = x[:, :, :D, :H, :W]

        return x



class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        # Define the encoding part (convolutions and downsampling)
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Define the bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Define the decoding part (upsampling and convolutions)
        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # Output layer
        self.output = nn.Conv3d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool3d(enc4, kernel_size=2, stride=2))

        # Decoding path
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output layer
        return self.output(dec1)



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)



class SimpleViT(nn.Module):
    def __init__(self, image_size, image_patch_size, slice_depth_size, slice_depth_patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert slice_depth_size % slice_depth_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (slice_depth_size // slice_depth_patch_size)
        patch_dim = channels * patch_height * patch_width * slice_depth_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = slice_depth_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)



    def forward(self, video):
        *_, h, w, dtype = *video.shape, video.dtype

        x = self.to_patch_embedding(video)
        pe = posemb_sincos_3d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        return x



class ProposedVnet(nn.Module):
    def __init__(self, image_size, slice_depth_size, image_patch_size, slice_depth_patch_size, dim, depth, heads,
                 mlp_dim, channels, dim_head, num_classes, survival_classes):
        super().__init__()
        self.image_size=image_size
        self.slice_depth_size=slice_depth_size
        self.image_patch_size=image_patch_size
        self.slice_depth_patch_size=slice_depth_patch_size
        self.numclasses = num_classes

        self.vit3d = SimpleViT(image_size=image_size, image_patch_size=image_patch_size, slice_depth_size=slice_depth_size,
                               slice_depth_patch_size=slice_depth_patch_size,
                               dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head)
        self.downconv = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm3d(dim),
                                      nn.ReLU(inplace=True))
        self.upconv = nn.Sequential(nn.ConvTranspose3d(dim, int(dim/2), kernel_size=4, stride=2, padding=1, output_padding=0),
                                    nn.BatchNorm3d(int(dim/2)),
                                    nn.ReLU(inplace=True))
        self.lastconv = nn.Sequential(nn.ConvTranspose3d(dim, num_classes, kernel_size=16, stride=8, padding=4, output_padding=0))
        self.onedownconv = nn.Sequential(nn.Conv3d(dim, int(dim/2), kernel_size=1, stride=1, padding=0),
                                         nn.BatchNorm3d(int(dim/2)),
                                         nn.ReLU(inplace=True))
        # Regression head for per-patch predictions
        self.regression_head = nn.Sequential(nn.Linear(dim, image_patch_size * image_patch_size * slice_depth_patch_size),
                                             nn.ReLU(inplace=True))


    def forward(self, x):
        t1 = self.vit3d(x)

        t1d = rearrange(t1, 'b (d h w) k -> b k d h w', d=int(self.slice_depth_size / self.slice_depth_patch_size),
                        h=int(self.image_size / self.image_patch_size))
        t1dc = self.downconv(t1d)
        t1dc = rearrange(t1dc, 'b k d h w -> b (d h w) k')

        t2 = self.vit3d.transformer(t1dc)

        t2u = rearrange(t2, 'b (d h w) k -> b k d h w',
                        d=int(self.slice_depth_size / (2 * self.slice_depth_patch_size)),
                        h=int(self.image_size / (2 * self.image_patch_size)))
        t2uc = self.upconv(t2u)
        t2ucat = torch.cat((self.onedownconv(t1d), t2uc), dim=1)
        t2ucat = rearrange(t2ucat, 'b k d h w -> b (d h w) k')

        t3 = self.vit3d.transformer(t2ucat)

        # t3u = rearrange(t3, 'b (d h w) k -> b k d h w', d=int(self.slice_depth_size / (self.slice_depth_patch_size)),
        #                 h=int(self.image_size / (self.image_patch_size)))
        out = self.regression_head(t3)
        return out

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3,
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16,
                 decoder_embed_dim=512, 
                 decoder_depth=8, 
                 decoder_num_heads=16,
                 mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False):

        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size 
        self.in_chans = in_chans

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.mask_func = {"random": self.random_masking}

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, 
                      m):
        
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adaptive_patchify(self,
                          imgs: torch.tensor,
                          patch_size = None) -> torch.tensor:
    
        """
        Split video into sequence of flattened patches
        (variably adapts to different number of 
        temporal dimensions represented as channels)
    
        Parameters
        ----------
        imgs: torch.Tensor
            A tensor of videos: (N, C, H, W)
    
        patch_size: int
            size of square patch to extract from videos 
    
        Returns
        -------
        x: torch.Tensor
            A tensor containing sequence of flattened patches 
            extracted from the videos 
            shape: (N, L, patch_size**2 *C)
        """
        if patch_size is None:
            patch_size = self.patch_size
    
        # check if input images are square by asserting height and
        # width equality, check if height is divisible by patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0
    
        # patches along height and width 
        h = w = imgs.shape[2] // patch_size
        
        # number of channels
        c = imgs.shape[1]
        
        # reshape tensor
        # splits the height and width into h and w patches of size p x p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, patch_size, w, patch_size))
        
        # permute tensor dimensions
            # n - (batch size) remains in place
            # c - (channels) moves to the last dimension
            # h - (patches along height) moves to the second dimension
            # w - (patches along width) moves to the third dimension
            # p (patch height) and q (patch width) are rearranged to be next to c
        x = torch.einsum('nchpwq->nhwpqc', x)
    
        # convert tensor into sequence of flattened patches
        x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * c))
        
        return x

    def adaptive_unpatchify(self,
                            patch_seq: torch.tensor,
                            patch_size = None) -> torch.tensor:
    
        """
        Restore video from a sequence of flattened
        patches based on patch_size
        (variably adapts to different numbers for temporal dimension
        represented as channels)
        
        Parameters
        ----------
        patch_seq: torch.Tensor
            Tensor of a sequence of flattened patches
            (N, L, patch_size**2 *C)
    
        patch_size: int
            size of square patch to restore video from
            input sequence of flattened patches
    
        Returns
        -------
        imgs: torch.Tensor
            A tensor of videos restored from
            their patch sequence format
            
            shape: (N, C, H, W)
        """
        if patch_size is None:
            patch_size = self.patch_size
            
        # number of patches along the height and width of the image
        h = w = int(patch_seq.shape[1]**.5)
        
        # verify that h * w = L 
        assert h * w == patch_seq.shape[1]
        
        # calculate number of channels in original image 
        c = patch_seq.shape[2]//(patch_size**2)
        
        # separate patches into a grid of shape h * w
        patch_seq = patch_seq.reshape(shape=(patch_seq.shape[0], h, w, patch_size, patch_size, c))
    
        # permute tensor dimensions 
            # n - Batch size (remains in the same position)
            # c - Number of channels (moves from the last position to the second position)
            # h - Number of patches along the height (moves from second to third position)
            # p - Height of each patch (remains in the same position)
            # w - Number of patches along the width (moves from the third to fifth position)
            # q - Width of each patch (moves from the fifth position to the sixth position)
        patch_seq = torch.einsum('nhwpqc->nchpwq', patch_seq)
    
        # reshape to original image 
        imgs = patch_seq.reshape(shape=(patch_seq.shape[0], c, h * patch_size, h * patch_size))
        
        return imgs

    def random_masking(self,
                       x: torch.tensor,
                       mask_ratio: float) -> Tuple[torch.tensor,
                                                   torch.tensor,
                                                   torch.tensor]:


        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  
        ids_shuffle = torch.argsort(noise, dim=1)  
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
    
        return x_masked, mask, ids_restore

    def forward_encoder(self,
                        imgs: torch.tensor,
                        mask_ratio: float,
                        mask_type: str = "random") -> Tuple[torch.tensor,
                                                            torch.tensor,
                                                            torch.tensor]:


        """
        Apply masking and forward-propagate images
        through encoder transformer blocks. 
        Encode to sequence of latent representations 
        of visible patch embeddings. 
    
        Parameters
        ----------
        imgs: torch.Tensor
            A tensor of images: (N, C, H, W)
    
        mask_ratio: float
            Percentage of patches to mask in sequence
            (masked patches are removed from sequence) 
    
        Returns
        -------
        Tuple[x,
              mask,
              ids_restore]
    
        x: torch.tensor
            Sequence of latent representations
            of visible patch embeddings
            shape: [N, int(L * (1 - mask_ratio)) + 1, D]
            where int(L * (1 - mask_ratio)) indicates the
            number of visible, unmasked patch tokens fed
            through encoder transformer blocks, including
            a cls_token 
            
            N - batch size
            L - sequence length (number of patches
            in original sequence of patches extracted
            from original image)
            D - patch embedding dimensionality (1-D vector)
    
        mask: torch.tensor
            binary mask representing patches that are
            masked or retained as visible
            0 is visible and 1 is masked
            shape: [N, L]
    
        ids_restore: torch.tensor
            indices that restore the original order
            of the patch sequence to which shuffling
            is applied in the function 
            shape: [N, L]
        """
    
        # convert image into sequence of patch embeddings
        # shape of x after patch embedding: (N, L, D)
        # N - batch size
        # L - number of patches in sequence (sequence length)
        # D - dimensionality of patch embedding (1-D vector)
        x = self.patch_embed(imgs)
    
        # add positional embeddings w/o cls token
        x = x + self.pos_embed[:, 1:, :]
    
        # apply masking, get masked sequence of visible
        # patch embeddings, binary mask and ids_restore 
                
        x, mask, ids_restore = self.mask_func[mask_type](x, mask_ratio)
        
        # append cls token
        # add positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    
        # concatenate cls_token to sequence
        x = torch.cat((cls_tokens, x), dim=1)
    
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
    
        # apply normalization
        x = self.norm(x)
    
        return x, mask, ids_restore

    def forward_decoder(self,
                        x: torch.tensor,
                        ids_restore: torch.tensor) -> torch.tensor:

        """
        Decode encoded sequence after decoder embedding,
        including mask tokens, and re-ordering sequence 
        back to original order of patches
    
        Parameters
        ----------
        
        x: torch.tensor
            Sequence of latent representations
            of visible patch embeddings (encoder output)
                
            shape: [N, int(L * (1 - mask_ratio)) + 1, D]
            where int(L * (1 - mask_ratio)) indicates the
            number of visible, unmasked patch tokens fed
            through encoder transformer blocks, including
            a cls_token 
                
            N - batch size
            L - sequence length (number of patches
                in original sequence of patches extracted
                from original image)
            D - patch embedding dimensionality (1-D vector)
            
    
        ids_restore: torch.tensor
            indices that restore the original order
            of the patch sequence to which shuffling
            is applied in the function 
            shape: [N, L]
    
        Returns
        -------
        x: torch.tensor
            Decoded sequence of patches
            including visible and mask tokens
            to reconstruct the full image
    
            shape: [N, L, patch_size**2 * C]
        """
    
        # embed tokens using linear projection
        # original dimension: [N, int(L * (1 - mask_ratio)) + 1, D]
        # transformed dimension: [N, int(L * (1 - mask_ratio)) + 1, decoder_embed_dim]
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        # repeat mask tokens and concatenate to input sequence of tokens
        # calculate number of mask tokens based on the difference the
        # original sequence length and the encoded sequence length post-masking
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        # concatenate x with mask_tokens, exclude cls_token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        
        # unshuffle the sequence
        # re-order the tokens in x_ according to its original order using ids_restore
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        # append cls_token
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # add positional embeddings
        x = x + self.decoder_pos_embed
        
        # apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        # apply normalization
        x = self.decoder_norm(x)
        
        # predictor projection
        # converts domensionality from
        # [N, L + 1, decoded_embed_dim]
        # to [N, L + 1, patch_size**2 * C]
        # C = in_chans and L = num_patches
        x = self.decoder_pred(x)
        
        # remove cls_token from final output
        # return only reconstructed image patches
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self,
                     imgs: torch.tensor,
                     pred: torch.tensor,
                     mask: torch.tensor,
                     visible_alpha: float = 0.0) -> torch.tensor:

        """
        Calculate forward_loss as a weighted
        sum of loss over masked patches and
        loss over visible patches
        Convert images into patches and
        calculates loss at the patch level
    
        Parameters
        ----------
        imgs: torch.tensor
            A tensor of images
            
            shape: (N, C, H, W)
    
        pred: torch.tensor
            Decoded sequence of patches
            including visible and mask tokens
            to reconstruct the full image
    
            shape: [N, L, patch_size**2 * C]
    
        mask: torch.tensor
            binary mask representing patches that are
            masked or retained as visible
            0 is visible and 1 is masked
            
            shape: [N, L]
    
        visible_alpha: float = 0.0 (default scalar)
            weight associated with loss over
            visible patches (visible_loss), 
            used to exponentially weigh mask_loss
            and visible_loss 
    
        Returns
        ----------
        whole_loss: torch.tensor (scalar)
            whole loss, calculated as exponentially weighted
            average between mean loss over masked (removed) patches
            and mean loss over visible (kept) patches
            
            whole_loss = ((1.0 - visible_alpha) * mask_loss) + (visible_alpha * visible_loss)

        mask_loss: torch.tensor (scalar)
            mean loss over masked (remove) patches

        visible_loss: torch.tensor (scalar)
            mean loss over visible (kept) patches 
        """
    
        # patchify images to serve as target
        # original imgs shape: [N, C, H, W]
        # target shape: [N, L, patch_size**2 * C]
        target = self.adaptive_patchify(imgs = imgs)
        
        # normalize target patches
        # mean is the mean of each patch (computed along the last dimension)
        # var is the variance of each patch (computed along the last dimension)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        # compute squared error
        # element-wise squared error between the predicted patches (pred) and the target patches (target)
        loss = (pred - target) ** 2
        
        # calculate mean loss per patch
        # shape: [N, L]
        loss = loss.mean(dim=-1)
        
        # calculate mean loss on removed (masked) patches
        # computed loss is element-wise multiplied by the mask
        # only the losses of the removed (masked) patches are considered
        mask_loss = (loss * mask).sum() / mask.sum()
        
        # invert mask to swap mask and visible patch positions
        # swap positions of 0s and 1s
        # 1 is remove (masked), 0 is keep in default mask
        inverse_mask = 1.0 - mask
        visible_loss = (loss * inverse_mask).sum() / inverse_mask.sum()
    
        # calculate whole loss as exponentially weighted average
        # of mask loss and visible loss
        whole_loss = ((1.0 - visible_alpha) * mask_loss) + (visible_alpha * visible_loss)
        
        return whole_loss, mask_loss, visible_loss

    def forward(self,
                imgs: torch.tensor, 
                mask_ratio: float = 0.0) -> Tuple[torch.tensor,
                                                     torch.tensor,
                                                     torch.tensor]:
    
        """
        MAE forward function. 
        Encode visible patches of input 
        images to latent sequence
        of visible patch embeddings, 
        generate mask, and ids_restore.
        Decode to predictions.
        Compute loss. 
    
        Parameters
        ----------
        imgs: torch.tensor
            A tensor of images
            
            shape: (N, C, H, W)
        
        mask_ratio: float
            Percentage of patches to mask in sequence
            (masked patches are removed from sequence
            and only the visible patches are encoded) 
            
        visible_alpha: float = 0.0 (default scalar)
            weight associated with loss over
            visible patches (visible_loss), 
            used to exponentially weigh mask_loss
            and visible_loss
    
        Returns
        ----------
        whole_loss: torch.tensor (scalar)
            Whole loss, calculated as exponentially weighted
            average between mean loss over masked (removed) patches
            and mean loss over visible (kept) patches
            
            whole_loss = ((1.0 - visible_alpha) * mask_loss) + (visible_alpha * visible_loss)

        mask_loss: torch.tensor (scalar)
            mean loss over masked (remove) patches

        visible_loss: torch.tensor (scalar)
            mean loss over visible (kept) patches 
    
        pred: torch.tensor
            Decoded sequence of patches
            including visible and mask tokens
            to reconstruct the full image
    
            shape: [N, L, patch_size**2 * C]
    
        mask: torch.tensor
            Binary mask representing patches 
            that are masked or retained as 
            visible
            
            0 is visible (kept) and 1 is masked (removed)
            shape: [N, L]
    
            N - batch size
            L - sequence length (number of patches
                in original sequence of patches extracted
                from original image)
        
        """
        imgs = imgs.squeeze(1)
        
        # encode to latent, mask and ids_restore
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    
        # decode to model predictions
        pred = self.forward_decoder(latent, ids_restore)
    
        # calculate loss
        #whole_loss, mask_loss, visible_loss = self.forward_loss(imgs, pred, mask, visible_alpha)
        
        out_vid = self.adaptive_unpatchify(pred)
        
        out_vid = out_vid.unsqueeze(1)
    
        return out_vid
