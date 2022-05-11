import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
from einops import rearrange


def imgs_to_patches(x, p):
    """
    Convert batch of imgs x w/ shape (b, c, h, w) to batch of patch sequences
    w/ shape (b, n, p^2 x c).
    """
    b, c, h, w = x.shape
    n = (h * w) // p**2

    x = x.unfold(2, p, p).unfold(3, p, p) # [b, c, n_h, n_w, p, p]
    x = x.permute(0, 2, 3, 1, 4, 5) # [b, n_h, n_w, c, p, p]
    x = x.flatten(start_dim=1)
    x = x.reshape(b, n, p**2 * c)
    return x


def shifted_patch_tokenization(x, p):
    """
    Shifted patch tokenization.
    """
    shift = p // 2
    x_pad = F.pad(x, (shift, shift, shift, shift))

    # shift in the four diagonal directions
    x_lu = x_pad[:, :, :-p, :-p]
    x_ru = x_pad[:, :, :-p, p:]
    x_lb = x_pad[:, :, p:, :-p]
    x_rb = x_pad[:, :, p:, p:]

    # concatenate the four shifted images with the input
    x_cat = torch.cat([x, x_lu, x_ru, x_lb, x_rb], dim=1)

    return x_cat


class LSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(self.head_dim ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, self.head_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(self.head_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class EncoderBlock(nn.Module):
    """
    Block class.

    Attributes:
        hidden_dim: latent vector size used through transformer layers
        mlp_dim: dimension of the mlp block on top of attention block
        num_heads: number of heads in multi-headed attention layers
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention layers
        lsa: use locality self-attention when true
    """
    def __init__(self, hidden_dim, mlp_dim, num_heads, dropout_rate, 
                 attention_dropout_rate, lsa):
        super(EncoderBlock, self).__init__()
        self.lsa = lsa

        self.ln1 = nn.LayerNorm(normalized_shape=hidden_dim)
        
        if lsa:
            self.attn = LSA(embed_dim=hidden_dim, num_heads=num_heads, 
                            dropout=attention_dropout_rate)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                            num_heads=num_heads, 
                                            dropout=attention_dropout_rate, 
                                            batch_first=True)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, inputs):
        x = self.ln1(inputs)

        if self.lsa:
            x = self.attn(x)
        else:
            x = self.attn(x, x, x)[0]

        # x = self.attn(x, x, x)[0]
        x = self.dropout(x)
        x = x + inputs # skip connection

        y = self.ln2(x)
        y = self.mlp(y)
        out = x + y # skip connection

        return out


class _VisionTransformer(nn.Module):
    """
    Model class.

    Attributes:
        input_dim: dimensions of input data
        patch_size: h / w of image patches
        num_layers: number of transformer encoder blocks 
        hidden_dim: latent vector size used through transformer layers
        mlp_dim: dimension of the mlp block on top of attention block
        num_heads: number of heads in multi-headed attention layers
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention layers
        num_classes: number of output classes
        spt: true if using shifted patch tokenization
        lsa: true if using locality self-attention
    """
    def __init__(self, input_dim, patch_size, num_classes, hidden_dim, mlp_dim, 
                 num_heads, num_layers, dropout_rate, attention_dropout_rate, 
                 spt, lsa):
        super().__init__()
        _, self.c, self.h, self.w = input_dim
        self.n = (self.h * self.w) // patch_size**2
        self.patch_size = patch_size
        self.spt = spt

        seq_len = patch_size**2 * self.c
        if spt:
            seq_len *= 5
        self.lin_proj = nn.Linear(seq_len, hidden_dim)
        
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n + 1, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.encoder = nn.Sequential(
            *[EncoderBlock(hidden_dim, mlp_dim, num_heads, dropout_rate, 
                           attention_dropout_rate, lsa) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_dim)

        self.classification_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        b = x.shape[0]

        # patch embedding
        if self.spt:
            x = shifted_patch_tokenization(x, p=self.patch_size)
        x = imgs_to_patches(x, p=self.patch_size)
        x = self.lin_proj(x)

        # learnable class token
        class_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        # positional encoding
        x = x + self.pos_embed

        # transformer encoder
        x = self.encoder(x)
        x = self.ln(x)

        # note: 'token' classification / TODO 'gap' classification
        x = x[:, 0]
        x = self.classification_head(x)

        return x


class VisionTransformer(pl.LightningModule):
    """
    Lightning wrapper class.
    """

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = _VisionTransformer(**model_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")