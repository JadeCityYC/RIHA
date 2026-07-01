import torch
import torch.nn as nn
import numpy as np

# Import multilevel_visual_extractor to replace the original visual_extractor
#from modules.visual_extractor_vit import VisualFeatureExtractor
from modules.multilevel_visual_extractor import VisualFeatureExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.alignment import MultiLevelAlignmentModule


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        
        # Use the new VisualFeatureExtractor to replace the original VisualExtractor
        self.visual_extractor = VisualFeatureExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        
        # Select the forward function based on the dataset
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
            
        # If alignment loss is enabled, create the alignment module
        if getattr(args, 'use_alignment_loss', False):
            # Set the text feature dimension
            text_dim = getattr(args, 'text_feature_dim', 768)
            
            # Create the alignment module
            self.alignment_module = MultiLevelAlignmentModule(
                visual_dims=(2048, 2048, 2048),  # visual feature dimensions
                text_dims=(text_dim, text_dim, text_dim),  # text feature dimensions
                projection_dim=getattr(args, 'projection_dim', 512),  # projection dimension
                alpha=getattr(args, 'low_level_weight', 0.3),
                beta=getattr(args, 'mid_level_weight', 0.4),
                gamma=getattr(args, 'high_level_weight', 0.3),
                ot_impl=getattr(args, 'ot_impl', 'pot-uot-l2'),
                ot_reg=getattr(args, 'ot_reg', 0.1),
                ot_tau=getattr(args, 'ot_tau', 0.5)
            )
            
            # Text feature attribute (will be set in set_text_features)
            self.text_features = None
        
    def set_text_features(self, text_features):
        """Set the text features used for alignment loss computation."""
        self.text_features = text_features

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', return_features=False):
        # Extract multi-level features for the first and second images
        features_0 = self.visual_extractor(images[:, 0])
        features_1 = self.visual_extractor(images[:, 1])
        
        # Use the fused features
        fused_feats_0 = features_0['fused_feats']  # [B, 49, 2048]
        fused_feats_1 = features_1['fused_feats']  # [B, 49, 2048]

        # Concatenate features from both images
        att_feats = torch.cat((fused_feats_0, fused_feats_1), dim=1)  # [B, 98, 2048]

        # For fc_feats, use the mean of each image's features
        fc_feats_0 = torch.mean(fused_feats_0, dim=1)  # [B, 2048]
        fc_feats_1 = torch.mean(fused_feats_1, dim=1)  # [B, 2048]
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)  # [B, 4096]

        # Combine multi-level features for alignment computation
        combined_features = {
            'low_level_feats': torch.cat((features_0['low_level_feats'], features_1['low_level_feats']), dim=1),
            'mid_level_feats': torch.cat((features_0['mid_level_feats'], features_1['mid_level_feats']), dim=1),
            'high_level_feats': torch.cat((features_0['high_level_feats'], features_1['high_level_feats']), dim=1),
            'highest_level_feats': torch.cat((features_0['highest_level_feats'], features_1['highest_level_feats']), dim=1),
            'fused_feats': att_feats
        }
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if return_features:
            return output, combined_features
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train', return_features=False):
        # Extract multi-level features from the image
        features = self.visual_extractor(images)

        # Use the fused features
        att_feats = features['fused_feats']  # [B, 49, 2048]

        # For fc_feats, use the mean of the features
        fc_feats = torch.mean(att_feats, dim=1)  # [B, 2048]
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if return_features:
            return output, features
        return output

