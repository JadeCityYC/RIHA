import torch
import torch.nn as nn
import torchvision.models as models


class VisualFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(VisualFeatureExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        
        # Use ResNet as the base model
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        # Extract different levels of features
        self.layer1 = nn.Sequential(*list(model.children())[:5])  # Low-level features
        self.layer2 = list(model.children())[5]  # Middle-level features
        self.layer3 = list(model.children())[6]  # Higher-level features
        self.layer4 = list(model.children())[7]  # Highest-level features

        # Convolutional layers to adjust feature dimensions
        self.low_conv = nn.Conv2d(256, 2048, kernel_size=1)
        self.mid_conv = nn.Conv2d(512, 2048, kernel_size=1)
        self.high_conv = nn.Conv2d(1024, 2048, kernel_size=1)
        self.highest_conv = nn.Conv2d(2048, 2048, kernel_size=1)

        # Feature fusion module
        self.fusion_weights = nn.Parameter(torch.FloatTensor([0.2, 0.2, 0.3, 0.3]))
        self.fusion_transform = nn.Linear(2048, 2048)
        self.fusion_norm = nn.LayerNorm(2048)

        # Upsampling layers to align feature sizes
        self.upsample = nn.Upsample(size=(7, 7), mode='nearest')

    def forward(self, images):
        # Ensure correct input format
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Extract features at different levels
        c1 = self.layer1(images)   # [B, 256, 56, 56]
        c2 = self.layer2(c1)       # [B, 512, 28, 28]
        c3 = self.layer3(c2)       # [B, 1024, 14, 14]
        c4 = self.layer4(c3)       # [B, 2048, 7, 7]

        # Adjust and upsample features to consistent size and dimension
        low_level_feats = self.low_conv(c1)   # [B, 2048, 56, 56]
        mid_level_feats = self.mid_conv(c2)   # [B, 2048, 28, 28]
        high_level_feats = self.high_conv(c3) # [B, 2048, 14, 14]
        highest_level_feats = self.highest_conv(c4)  # [B, 2048, 7, 7]

        # Upsample all features to 7x7
        low_level_feats = self.upsample(low_level_feats)
        mid_level_feats = self.upsample(mid_level_feats)
        high_level_feats = self.upsample(high_level_feats)

        # Normalize fusion weights
        weights = torch.softmax(self.fusion_weights, dim=0)

        # Weighted fusion of features
        fused_feats = (
            weights[0] * low_level_feats + 
            weights[1] * mid_level_feats + 
            weights[2] * highest_level_feats 
            #weights[3] * highest_level_feats
        )  # [B, 2048, 7, 7]

        # Reshape features to patch format
        batch_size = fused_feats.size(0)
        
        # Reshape individual level features
        low_level_patches = low_level_feats.view(batch_size, 2048, -1).permute(0, 2, 1)    # [B, 49, 2048]
        mid_level_patches = mid_level_feats.view(batch_size, 2048, -1).permute(0, 2, 1)    # [B, 49, 2048]
        high_level_patches = high_level_feats.view(batch_size, 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]
        highest_level_patches = highest_level_feats.view(batch_size, 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]
        
        # Reshape fused features
        fused_patches = fused_feats.view(batch_size, 2048, -1).permute(0, 2, 1)  # [B, 49, 2048]

        # Apply transformation and normalization to fused features
        fused_patches = self.fusion_norm(self.fusion_transform(fused_patches))

        return {
            'low_level_feats': low_level_patches,
            'mid_level_feats': mid_level_patches,
            'high_level_feats': high_level_patches,
            'highest_level_feats': highest_level_patches,
            'fused_feats': fused_patches
        }