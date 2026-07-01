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
    Compute the total loss, including the language model loss and an optional alignment loss.

    Args:
        output: model output (used for language model loss)
        reports_ids: report IDs (used for language model loss)
        reports_masks: report masks (used for language model loss)
        visual_features: optional, visual features (used for alignment loss)
        text_features: optional, text features (used for alignment loss)
        report_ids: optional, list of report IDs in the current batch (used for alignment loss)
        alignment_module: optional, MultiLevelAlignmentModule instance
        alignment_weight: weight coefficient for alignment loss, default 0.0 (no alignment loss)

    Returns:
        torch.Tensor: total loss
        dict: detailed breakdown of individual losses (when alignment loss is used)
    """
    # Compute the language model loss
    criterion = LanguageModelCriterion()
    lm_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    
    # Initialize the loss details dictionary
    loss_details = {'lm_loss': lm_loss.item()}
    
    # Compute alignment loss if all required components are provided
    if (visual_features is not None and text_features is not None and 
        report_ids is not None and alignment_module is not None and 
        alignment_weight > 0):
        
        # Compute alignment loss
        alignment_loss, align_details = alignment_module(
            visual_features, text_features, report_ids)
        
        # Combine losses
        total_loss = lm_loss + alignment_weight * alignment_loss
        
        # Update loss details
        loss_details.update(align_details)
        loss_details['alignment_loss'] = alignment_loss.item()
        loss_details['total_loss'] = total_loss.item()
        
        return total_loss, loss_details
    
    # No alignment loss: return only the language model loss
    return lm_loss, loss_details