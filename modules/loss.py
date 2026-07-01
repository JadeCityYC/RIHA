import torch
import torch.nn as nn
from modules.alignment import MultiLevelAlignmentModule


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks, 
                visual_features=None, text_features=None, report_ids=None, 
                alignment_module=None, alignment_weight=0.0):
    """
    计算总的损失，包括语言模型损失和可选的对齐损失
    
    Args:
        output: 模型的输出 (用于语言模型损失)
        reports_ids: 报告ID (用于语言模型损失)
        reports_masks: 报告掩码 (用于语言模型损失)
        visual_features: 可选，视觉特征 (用于对齐损失)
        text_features: 可选，文本特征 (用于对齐损失)
        report_ids: 可选，当前批次的报告ID列表 (用于对齐损失)
        alignment_module: 可选，多级别对齐模块实例
        alignment_weight: 对齐损失的权重系数，默认为0.0 (不使用对齐损失)
        
    Returns:
        torch.Tensor: 总损失
        dict: 包含各种损失的详细信息 (如果有对齐损失)
    """
    # 计算语言模型损失
    criterion = LanguageModelCriterion()
    lm_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    
    # 初始化损失详情字典
    loss_details = {'lm_loss': lm_loss.item()}
    
    # 如果提供了必要组件，计算对齐损失
    if (visual_features is not None and text_features is not None and 
        report_ids is not None and alignment_module is not None and 
        alignment_weight > 0):
        
        # 计算对齐损失
        alignment_loss, align_details = alignment_module(
            visual_features, text_features, report_ids)
        
        # 合并损失
        total_loss = lm_loss + alignment_weight * alignment_loss
        
        # 更新损失详情
        loss_details.update(align_details)
        loss_details['alignment_loss'] = alignment_loss.item()
        loss_details['total_loss'] = total_loss.item()
        
        return total_loss, loss_details
    
    # 如果没有对齐损失，只返回语言模型损失
    return lm_loss, loss_details