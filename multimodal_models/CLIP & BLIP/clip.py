from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int,    # dimension of each row (column size) 
                  n_head: int,
                    attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # remain 
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        '''  
        x: (number_of_patches+1, Num batches, width)
        '''
        # build attention mask so that it automatically filters out 
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        #print(f"size of x in attention: {x.shape}")
        # pass x itself for Q, K, V, which is perfrming slef attention
        # output the same dimension 
        att= self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

        #print(f"Shape of attention after transformer output: {att.shape}")
        return att 

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))  # projection + layernorm 
        return x


class Transformer(nn.Module):
    def __init__(self, width: int,  # number of RGB channels 
                 layers: int,   # number of multi-attention head layers
                 heads: int,    # number of heads 
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    '''  
    process input iamge
    
    '''
    def __init__(self, input_resolution: int,   # With or height (assume square image)
                  patch_size: int,  # the kernel size 
                    width: int,         # number of channels of the image to project to, here we keep it as constant for all layers during processing
                    layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim    # output embedding dimension 

        # width specifies the number of channels of image output 
        # since the image is divided into patches, we use kernel_size as patch size to divide up the input image 
        # convert into H/patch, W/patch 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # 1/ sqrt(output_channel)   scaling 
        scale = width ** -0.5

        ''' 
        class_embedding: embed the entire patch of pixels into a single vector 
                - initialize as random numbers for each width 

        positional_embedding: add additional position information to each patch before processing, because transformers won't understand positional differences 
            - these positional embedding will provide information
            - extra positional embedding for each of the class_embedding value 
            - (Total number of token+1, width)
        '''
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        # this is also the convolution calculation formula, which is the equivalent to conv1 
        # applied to all the values lined into a single vector after convolution
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # transformer head to process 
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)

        # manual matrix multiplication, not standard feedforward neural network
        # no bias added 
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):

        # make into paches 
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        #print(f"Shape after conv: {x.shape}")

        # flatten each of the RGB dimension into single vector linearly 
        ''' 
        [               =>   [+,+,+,-,-,-]
        [+,+,+],
        [-,-,-] 
        ]
        '''
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        
        # flatten the patches 
        # shape = [*, grid ** 2, width]
        x = x.permute(0, 2, 1)  
       
        # add class_embedding
        ''' 
        images are passed in as regular N x C x H x W size 
        convolution layers are used to break into patches with extra rgb dimension 
        each class_embedding serves as summary to the entire image RGB dimension 
        (N, total_numer_of_patches+1, width)
        
        '''
        # shape = [*, grid ** 2 + 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) 
        
        # (N,patches+1, width)
        x = x + self.positional_embedding.to(x.dtype)   
       
       
        #print(f"Shape after positioning: {x.shape}")    

        # layer norm 
        x = self.ln_pre(x)

        # change dimension for parallelization 
        # (pachtes+1, N, width)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        

        # process back into 
        # (N, patches+1, width) 
        x = x.permute(1, 0, 2)  # LND -> NLD

        #print(f"Shape after transformer: {x.shape}")

        # layer norm with respect to the summary token 
        # only take the first row of summary token, then apply norm to the first
        x = self.ln_post(x[:, 0, :])
        #print(f"Shape after layer norm: {x.shape}")

        # output embedding dimension 
        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int ,   # the output embedding dimension (D_i = D_t)
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        # check if vision layers is passed in the specific dimension size 
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )

        # build vision transformer 
        else:
            # divide by 64 to get the number of heads needed to get that amount of vision width 
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # learnable temperature parameter applied before softmax 
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        '''  
        text: raw text in one hot encoding 
        '''

        # conver to hidden dimension 
        # each sentence would be turned into the same length (sequence_len)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        #print(f"Size of text x after token embedding: {x.shape}")
        #print(f"x after embedding: {x}")

        # add positional encoding with learnable parameters 
        x = x + self.positional_embedding.type(self.dtype)
        
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        #print(f"x after transformer: {x}")

        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # finds EOS token location for each batch, and projects it 
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        '''
        image: N x (dimensions of given image) 
        text: N x (dimensions of embedding) 
        '''


        # first encode the image with ResNet or VisionTransformer 
        # (NxD)
        ''' 
        the images is divided into patches at each dimension, each dimension has one summary token 
        the tokens go through the same kind of attention to learn each other's embedding 
        '''
        image_features = self.encode_image(image)

        # encode the text with the regular Transformer (Torch.Tensor)
        text_features = self.encode_text(text)

        # normalized features to prevent feature out-of-scale 
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # logit scales, turn into temperature 
        logit_scale = self.logit_scale.exp()

        # logits of images, logits allow values to be more split/certain 
        
        # finds how close an image is to a certain text (image to text)
        logits_per_image = logit_scale * image_features @ text_features.t()

        # how close a text is to which image (text to image) 
        logits_per_text = logits_per_image.t()
        
        ''' 
        [imgTok1, img1Tok2, img1Tok3, img1Tok4]     @ [txt1Tok1, txt2Tok1]
        [img2Tok1, img2Tok2, img2Tok3, img2Tok4]       [txt1Tok2, txt2Tok2]
                                                        [txt1Tok3, txt2Tok3]
                                                        [txt1Tok4, txt2Tok4]

        output of logits_per_image  (img to text)
                txt1    txt2
        img1    4.6     2.1
        img2    1.8     10.5
        ... 

        logits_per_text (compare text to which image has the highest value)
                img1    img2 
        txt1    4.6     1.8 
        txt2    2.1     10.5

        diagonal stays the same 
        '''

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

def loss_function(logits_image, logits_text): 
    '''  
    loss function is defined as cross entropy, to maximize the probability of choosing the closest match pair
    the ground truth is always the diagonal, by data design, they match each other 
    '''
    # assure both image and text have the same number of samples 
    assert(logits_image.shape[0], logits_text.shape[0])

    # define ground trueh 
    N = logits_image.shape[0]
    labels= torch.arange(N, dtype=torch.float32) # send to same device, make sure of floating size 

    # go across each image row and calculate the loss with each text 
    loss_image= F.cross_entropy(logits_image, labels)
    
    # go across each text and look at which image has the highest match, should be the same 
    loss_text= F.cross_entropy(logits_text, labels)

    # sum the mean loss (2*|B|), we take the average for the best comparison
    # in the future, we can potentially try weighted, so that certian loss of image to text or text to image is better 
    total= (loss_image+ loss_text)/ (2)    #(N,1)

    return total 