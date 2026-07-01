import torch
import torch.nn as nn
import numpy as np

# 导入multilevel_visual_extractor替代原来的visual_extractor
#from modules.visual_extractor_vit import VisualFeatureExtractor
from modules.multilevel_visual_extractor import VisualFeatureExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.alignment import MultiLevelAlignmentModule


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        
        # 使用新的VisualFeatureExtractor替代原来的VisualExtractor
        self.visual_extractor = VisualFeatureExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        
        # 根据数据集选择前向传播函数
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
            
        # 如果启用了对齐损失，创建对齐模块
        if getattr(args, 'use_alignment_loss', False):
            # 设置文本特征维度
            text_dim = getattr(args, 'text_feature_dim', 768)
            
            # 创建对齐模块
            self.alignment_module = MultiLevelAlignmentModule(
                visual_dims=(2048, 2048, 2048),  # 视觉特征维度
                text_dims=(text_dim, text_dim, text_dim),  # 文本特征维度
                projection_dim=getattr(args, 'projection_dim', 512),  # 投影维度
                alpha=getattr(args, 'low_level_weight', 0.3),
                beta=getattr(args, 'mid_level_weight', 0.4),
                gamma=getattr(args, 'high_level_weight', 0.3),
                ot_impl=getattr(args, 'ot_impl', 'pot-uot-l2'),
                ot_reg=getattr(args, 'ot_reg', 0.1),
                ot_tau=getattr(args, 'ot_tau', 0.5)
            )
            
            # 文本特征属性（将在set_text_features中设置）
            self.text_features = None
        
    def set_text_features(self, text_features):
        """设置用于对齐损失计算的文本特征"""
        self.text_features = text_features

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', return_features=False):
        # 获取第一张和第二张图像的多级别特征
        features_0 = self.visual_extractor(images[:, 0])
        features_1 = self.visual_extractor(images[:, 1])
        
        # 使用融合后的特征
        fused_feats_0 = features_0['fused_feats']  # [B, 49, 2048]
        fused_feats_1 = features_1['fused_feats']  # [B, 49, 2048]
        
        # 合并两张图像的特征
        att_feats = torch.cat((fused_feats_0, fused_feats_1), dim=1)  # [B, 98, 2048]
        
        # 对于fc_feats，我们可以使用每张图像特征的平均值
        fc_feats_0 = torch.mean(fused_feats_0, dim=1)  # [B, 2048]
        fc_feats_1 = torch.mean(fused_feats_1, dim=1)  # [B, 2048]
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)  # [B, 4096]
        
        # 合并多级别特征用于对齐计算
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
            raise ValueError(f"不支持的模式: {mode}")
        
        if return_features:
            return output, combined_features
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train', return_features=False):
        # 获取图像的多级别特征
        features = self.visual_extractor(images)
        
        # 使用融合后的特征
        att_feats = features['fused_feats']  # [B, 49, 2048]
        
        # 对于fc_feats，我们使用特征的平均值
        fc_feats = torch.mean(att_feats, dim=1)  # [B, 2048]
        
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        if return_features:
            return output, features
        return output

