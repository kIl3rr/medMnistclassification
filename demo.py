import torch 
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import collections.abc
import cv2

# Define constants
num_classes = 6
input_shape = (32, 32, 1)  # Grayscale, so channel is 1
patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 128  # MLP layer size
# Convert embedded patches to query, key, and values with a learnable additive
# value
qkv_bias = True
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 32  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 128
num_epochs = 2
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.5

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_window_elements, num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_matrix = torch.meshgrid(coords_h, coords_w)
        coords = torch.stack(coords_matrix)
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))

    def forward(self, x, mask=None):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = self.relative_position_index.view(-1)
        relative_position_bias = self.relative_position_bias_table[relative_position_index_flat].view(
            num_window_elements, num_window_elements, -1
        ).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn += mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x_qkv = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_qkv = self.proj(x_qkv)
        return x_qkv

class PatchMerging(nn.Module):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)

    def forward(self, x):
        batch_size, numt, C = x.shape
        height,width = self.num_patch
        # batch_size, height, width, C = x.size()
        x = x.view(batch_size, height, width, C)
        # print("initial_patch_merging:", x.size())
        x = x.permute(0, 3, 1, 2)  # Change dimensions to (batch_size, C, height, width)
        # print("after_permute:", x.size())
        x = x.reshape(batch_size, C, height // 2, 2, width // 2, 2)
        # print("after_reshape:", x.size())
        x = x.permute(0, 1, 3, 5, 2, 4)
        # print("after_permute:", x.size())
        x = x.reshape(batch_size, -1, height // 2, width // 2)
        # print("after_reshape:", x.size())
        x = x.permute(0, 2, 3, 1)  # Change dimensions back to (batch_size, height // 2, width // 2, ...)
        # print("after_permute:", x.size())
        x = x.reshape(batch_size, -1, 4 * C)
        # print("after_reshape:", x.size())
        temp = self.linear_trans(x)
        # print("temp_size:", temp.size())
        return temp



def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # print(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # print("H:", H)
        # print("W:", W)
        # print("patch_embedding_x_shape",x.shape)
        # print("hello")
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = x + self.pos_embed
        if self.norm is not None:
            x = self.norm(x)
        return x





def window_partition(x, window_size):
    B, H, W, C = x.shape
    # print("window_partition_x_shape:", x.shape)
    patch_num_y = H // window_size
    patch_num_x = W // window_size
    x = x.view(B, patch_num_y, window_size, patch_num_x, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    windows = x.reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = windows.view(-1, patch_num_y, patch_num_x, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(-1, height, width, channels)
    return x

class LayerNormalization(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class SwinTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
    ):
        super(SwinTransformer, self).__init__()

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp

        self.norm1 = LayerNormalization(dim)
        self.attn = WindowAttention(dim, window_size =(window_size,window_size), num_heads = num_heads, qkv_bias = qkv_bias, dropout_rate = dropout_rate)
        self.drop_path = nn.Dropout(dropout_rate)
        self.norm2 = LayerNormalization(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, num_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_mlp, dim),
            nn.Dropout(dropout_rate)
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def forward(self, x):
        height, width = self.num_patch
        batch_size, num_patches_before, channels = x.shape
        # print("swin_layer_x_batch:", x.shape)
        x_skip = x.clone()
        x = self.norm1(x)
        x = x.view(batch_size, height, width, channels)
        
        if self.shift_size > 0:
            shifted_x = torch.cat((x[:, :, self.shift_size:, :], x[:, :, :self.shift_size, :]), dim=2)
            shifted_x = torch.cat((shifted_x[:, :, :, self.shift_size:], shifted_x[:, :, :, :self.shift_size]), dim=3)
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width, channels)
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch_size, height * width, channels)
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        # print("swin_layer_x_shape:", x.shape)
        return x

class SwinModel(nn.Module):
    def __init__(self, embed_dim, num_patch_x, num_patch_y, num_heads, num_mlp, window_size, shift_size, qkv_bias, num_classes):
        super(SwinModel, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size=(32,32), patch_size=(2,2), embed_dim=64)
        # PatchEmbedding(image_size=(32,32), patch_size=(2,2), embed_dim=64)
        self.swin_transformer1 = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=0.0
        )
        self.swin_transformer2 = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=0.0
        )
        self.patch_merging = PatchMerging((num_patch_x, num_patch_y), embed_dim)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
#         print("swin_model_x_shape:", x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.patch_embedding(x)
        x = self.swin_transformer1(x)
        x = self.swin_transformer2(x)
        x = self.patch_merging(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.softmax(x)
#         print("after_fc_swin_model_x_shape:", x.shape)
        return x


model = SwinModel(embed_dim, num_patch_x, num_patch_y, num_heads, num_mlp, window_size, shift_size, qkv_bias, num_classes)
model.load_state_dict(torch.load('./model/swin_classification_pytorch_model_weights.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到之前训练时使用的设备device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()  # 设置为评估模式

# 加载图像，调整大小，归一化
image = cv2.imread('./MedMNIST/CXR/006321.jpeg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (32, 32))
image = image.astype(np.float32) / 255.0  # 归一化

# 将图像数据调整到正确的维度
image = np.expand_dims(image,axis=1)  # 确保形状是 (32, 32, 1)
image = np.transpose(image, (2, 0, 1))  # 重新排列为 (C, H, W)
image = np.expand_dims(image, axis=0)  # 添加批次维度，变成 (N, C, H, W)

# 转换为张量
image_tensor = torch.from_numpy(image)

# 确保模型在正确的设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_tensor = image_tensor.to(device)

name = {
    0: "手",
    1: "头部",
    2: "肺部",
    3: "??",
    4: "",
    5: "手"
}

# 进行预测
model.eval()
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted label: {predicted.item()}')


