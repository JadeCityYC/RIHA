import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn
import ot
import numpy as np


class OT_Attn_assem(nn.Module):
    def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)
    
    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Compute optimal transport between two sets of features.

        Args:
            weight1 : (N, D) visual features
            weight2 : (M, D) text features

        Return:
            flow : (N, M) transport flow matrix
            dist : (1, ) transport distance value
        """
        # Ensure inputs are 2D tensors
        if weight1.dim() == 1:
            weight1 = weight1.unsqueeze(0)  # Add batch dimension
        if weight2.dim() == 1:
            weight2 = weight2.unsqueeze(0)  # Add batch dimension

        # If features are empty (some dimension is 0), return zero distance
        if weight1.size(0) == 0 or weight2.size(0) == 0:
            return torch.zeros((1, 1), device=weight1.device), torch.tensor(0.0, device=weight1.device)

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2  # (N, M)
            
            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow 
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a = torch.from_numpy(ot.unif(weight1.size(0)).astype('float64')).to(weight1.device)
            b = torch.from_numpy(ot.unif(weight2.size(0)).astype('float64')).to(weight2.device)
            
            self.cost_map = torch.cdist(weight1, weight2)**2  # (N, M)
            
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double(), reg=self.ot_reg, reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).to(weight1.device)  # Cast to float and move to the same device
            
            dist = self.cost_map * flow  # (N, M)
            dist = torch.sum(dist)  # (1,) scalar float
            return flow, dist
        
        else:
            raise NotImplementedError
    # def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
    #     super().__init__()
    #     self.impl = impl
    #     self.ot_reg = ot_reg
    #     self.ot_tau = ot_tau
    #     print("ot impl: ", impl)
    
    # def normalize_feature(self,x):
    #     x = x - x.min(-1)[0].unsqueeze(-1)
    #     return x

    # def OT(self, weight1, weight2):
    #     """
    #     Parmas:
    #         weight1 : (N, D)
    #         weight2 : (M, D)
        
    #     Return:
    #         flow : (N, M)
    #         dist : (1, )
    #     """

    #     if self.impl == "pot-sinkhorn-l2":
    #         self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
    #         src_weight = weight1.sum(dim=1) / weight1.sum()
    #         dst_weight = weight2.sum(dim=1) / weight2.sum()
            
    #         cost_map_detach = self.cost_map.detach()
    #         flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
    #                             M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
    #         dist = self.cost_map * flow 
    #         dist = torch.sum(dist)
    #         return flow, dist
        
    #     elif self.impl == "pot-uot-l2":
    #         a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)
    #         self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
    #         cost_map_detach = self.cost_map.detach()
    #         M_cost = cost_map_detach/cost_map_detach.max()
            
    #         flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
    #                             M=M_cost.double(), reg=self.ot_reg,reg_m=self.ot_tau)
    #         flow = flow.type(torch.FloatTensor).cuda()
            
    #         dist = self.cost_map * flow # (N, M)
    #         dist = torch.sum(dist) # (1,) float
    #         return flow, dist
        
    #     else:
    #         raise NotImplementedError

        

    def forward(self,x,y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()
        
        x = self.normalize_feature(x)
        y = self.normalize_feature(y)
        
        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist

class MultiLevelAlignmentModule(nn.Module):
    """
    Multi-level feature alignment module with learnable projection layers.
    """
    def __init__(self, visual_dims=(2048, 2048, 2048), 
                 text_dims=(768, 768, 768),
                 projection_dim=512,
                 alpha=0.4, beta=0.2, gamma=0.4,
                 ot_impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5):
        """
        Initialize the multi-level feature alignment module.

        Args:
            visual_dims: tuple of visual feature dimensions (low-level dim, mid-level dim, high-level dim)
            text_dims: tuple of text feature dimensions (word dim, sentence dim, paragraph dim)
            projection_dim: shared projection dimension after projection
            alpha: loss weight for low-level (visual-word) alignment
            beta: loss weight for mid-level (visual-sentence) alignment
            gamma: loss weight for high-level (visual-paragraph) alignment
            ot_impl: OT implementation method
            ot_reg: OT regularization parameter
            ot_tau: UOT marginal relaxation parameter
        """
        super().__init__()
        
        self.projection_dim = projection_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Create OT calculator
        self.ot_calculator = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

        # Create projection layers - low-level (word-level)
        self.low_visual_projection = nn.Linear(visual_dims[0], projection_dim)
        self.word_projection = nn.Linear(text_dims[0], projection_dim)

        # Create projection layers - mid-level (sentence-level)
        self.mid_visual_projection = nn.Linear(visual_dims[1], projection_dim)
        self.sentence_projection = nn.Linear(text_dims[1], projection_dim)

        # Create projection layers - high-level (paragraph-level)
        self.high_visual_projection = nn.Linear(visual_dims[2], projection_dim)
        self.paragraph_projection = nn.Linear(text_dims[2], projection_dim)

        # Initialize projection layer weights
        self._init_projection_weights()
    
    def _init_projection_weights(self):
        """Initialize projection layer weights."""
        for module in [self.low_visual_projection, self.word_projection,
                       self.mid_visual_projection, self.sentence_projection,
                       self.high_visual_projection, self.paragraph_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    

    def convert_dict_to_tensor(self, feature_dict, ids_list, device):
        """Convert a feature dictionary to a tensor."""
        features = []
        valid_ids = []

        # Collect all valid features first
        for id_key in ids_list:
            if id_key in feature_dict:
                # If numpy array or list, convert to tensor
                if isinstance(feature_dict[id_key], (list, np.ndarray)):
                    # Check that the feature is non-empty
                    if len(feature_dict[id_key]) > 0:
                        features.append(torch.tensor(feature_dict[id_key], dtype=torch.float).to(device))
                        valid_ids.append(id_key)
                # If already a tensor, ensure it is on the correct device
                elif isinstance(feature_dict[id_key], torch.Tensor):
                    # Check that the feature is non-empty
                    if feature_dict[id_key].numel() > 0:
                        features.append(feature_dict[id_key].to(device))
                        valid_ids.append(id_key)

        # If no valid features were collected, return None
        if not features:
            return None

        # Check that all features share the same last dimension
        feat_dim = features[0].size(-1)
        for i, feat in enumerate(features):
            if feat.size(-1) != feat_dim:
                print(f"Warning: feature dimension mismatch! ID {valid_ids[i]} has dim {feat.size(-1)}, expected {feat_dim}")
                # Adjust dimension (truncate or pad)
                if feat.size(-1) > feat_dim:
                    features[i] = feat[..., :feat_dim]
                else:
                    pad_size = feat_dim - feat.size(-1)
                    features[i] = torch.cat([feat, torch.zeros((*feat.shape[:-1], pad_size), device=device)], dim=-1)

        # Attempt to stack features; handle mismatched shapes if necessary
        try:
            return torch.stack(features)
        except RuntimeError as e:
            print(f"Error stacking features: {e}")
            print(f"Feature shapes: {[f.shape for f in features]}")

            # Attempt to pad all features to the same shape
            max_dims = [max(f.size(i) for f in features) for i in range(features[0].dim())]
            adjusted_features = []

            for feat in features:
                if feat.dim() != len(max_dims):
                    print(f"Warning: feature has different number of dimensions {feat.dim()} vs {len(max_dims)}")
                    continue

                # Adjust each dimension
                current_shape = list(feat.shape)
                needs_padding = any(current_shape[i] != max_dims[i] for i in range(len(current_shape)))
                
                if needs_padding:
                    padded_feat = feat
                    for dim, (current, target) in enumerate(zip(current_shape, max_dims)):
                        if current < target:
                            pad_size = target - current
                            padding = [0] * (2 * feat.dim())
                            padding[2 * dim + 1] = pad_size
                            padded_feat = torch.nn.functional.pad(padded_feat, tuple(reversed(padding)))
                    adjusted_features.append(padded_feat)
                else:
                    adjusted_features.append(feat)
            
            if adjusted_features:
                return torch.stack(adjusted_features)
            return None
    # def convert_dict_to_tensor(self, feature_dict, ids_list, device):
    #     """Convert a feature dictionary to a tensor."""
    #     features = []
    #     for id_key in ids_list:
    #         if id_key in feature_dict:
    #             # If numpy array, convert to tensor
    #             if isinstance(feature_dict[id_key], (list, np.ndarray)):
    #                 features.append(torch.tensor(feature_dict[id_key], dtype=torch.float).to(device))
    #             # If already a tensor, ensure it is on the correct device
    #             elif isinstance(feature_dict[id_key], torch.Tensor):
    #                 features.append(feature_dict[id_key].to(device))
        
    #     if features:
    #         return torch.stack(features)
    #     else:
    #         return None
    
    # 
    def calculate_ot_distance(self, visual_features, text_features):
        """Compute the OT distance between projected features."""
        # Check if features are empty
        if text_features is None or visual_features.size(0) == 0:
            return torch.tensor(0.0, device=visual_features.device)

        # Check text feature dimensionality and size
        if text_features.dim() == 0 or text_features.size(0) == 0:
            return torch.tensor(0.0, device=visual_features.device)

        # Ensure features have the correct number of dimensions
        if visual_features.dim() == 1:
            visual_features = visual_features.unsqueeze(0)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)

        try:
            _, dist = self.ot_calculator(visual_features.unsqueeze(1), text_features.unsqueeze(1))
            # Normalize the distance
            normalized_dist = dist / (visual_features.size(0) * text_features.size(0))
            return normalized_dist
        except RuntimeError as e:
            print(f"Error computing OT distance: {e}")
            print(f"Visual feature shape: {visual_features.shape}, text feature shape: {text_features.shape}")
            return torch.tensor(0.0, device=visual_features.device)
    
    def compute_low_level_alignment(self, low_visual_features, word_features, report_id):
        """Compute low-level alignment between visual low-level features and word features."""
        device = low_visual_features.device

        # Retrieve word IDs associated with the report
        word_ids = [f"{report_id}_w{j}" for j in range(100)]  # Assume at most 100 words
        word_tensor = self.convert_dict_to_tensor(word_features, word_ids, device)

        if word_tensor is not None and word_tensor.size(0) > 0:
            # Project features into the shared space
            projected_visual = self.low_visual_projection(low_visual_features)
            projected_text = self.word_projection(word_tensor)

            # Compute distance
            low_word_loss = self.calculate_ot_distance(projected_visual, projected_text)
        else:
            low_word_loss = torch.tensor(0.0, device=device)

        return low_word_loss
    
    def compute_mid_level_alignment(self, mid_visual_features, sentence_features, report_id):
        """Compute mid-level alignment between visual mid-level features and sentence features."""
        device = mid_visual_features.device

        # Retrieve sentence IDs associated with the report
        sentence_ids = [f"{report_id}_s{j}" for j in range(20)]  # Assume at most 20 sentences
        sentence_tensor = self.convert_dict_to_tensor(sentence_features, sentence_ids, device)

        if sentence_tensor is not None and sentence_tensor.size(0) > 0:
            # Project features into the shared space
            projected_visual = self.mid_visual_projection(mid_visual_features)
            projected_text = self.sentence_projection(sentence_tensor)

            # Compute distance
            mid_sentence_loss = self.calculate_ot_distance(projected_visual, projected_text)
        else:
            mid_sentence_loss = torch.tensor(0.0, device=device)

        return mid_sentence_loss
    
    def compute_high_level_alignment(self, high_visual_features, paragraph_features, report_id):
        """Compute high-level alignment between visual high-level features and paragraph features."""
        device = high_visual_features.device

        # Retrieve the paragraph feature
        if report_id in paragraph_features:
            # Convert paragraph feature to tensor
            if isinstance(paragraph_features[report_id], (list, np.ndarray)):
                paragraph_tensor = torch.tensor(paragraph_features[report_id], dtype=torch.float).to(device).unsqueeze(0)
            else:
                paragraph_tensor = paragraph_features[report_id].to(device).unsqueeze(0)

            # If the feature is empty, return zero loss
            if paragraph_tensor.numel() == 0:
                return torch.tensor(0.0, device=device)

            # Repeat paragraph feature to match the number of visual features
            paragraph_tensor = paragraph_tensor.repeat(high_visual_features.size(0), 1)

            # Project features into the shared space
            projected_visual = self.high_visual_projection(high_visual_features)
            projected_text = self.paragraph_projection(paragraph_tensor)

            # Compute distance
            high_paragraph_loss = self.calculate_ot_distance(projected_visual, projected_text)
        else:
            high_paragraph_loss = torch.tensor(0.0, device=device)

        return high_paragraph_loss
    
    # def compute_low_level_alignment(self, low_visual_features, word_features, report_id):
    #     """Compute low-level alignment between visual low-level features and word features."""
    #     device = low_visual_features.device

    #     # Retrieve word IDs associated with the report
    #     word_ids = [f"{report_id}_w{j}" for j in range(100)]  # Assume at most 100 words
    #     word_tensor = self.convert_dict_to_tensor(word_features, word_ids, device)

    #     if word_tensor is not None:
    #         # Project features into the shared space
    #         projected_visual = self.low_visual_projection(low_visual_features)
    #         projected_text = self.word_projection(word_tensor)

    #         # Compute distance
    #         low_word_loss = self.calculate_ot_distance(projected_visual, projected_text)
    #     else:
    #         low_word_loss = torch.tensor(0.0, device=device)

    #     return low_word_loss
    
    # def compute_mid_level_alignment(self, mid_visual_features, sentence_features, report_id):
    #     """Compute mid-level alignment between visual mid-level features and sentence features."""
    #     device = mid_visual_features.device

    #     # Retrieve sentence IDs associated with the report
    #     sentence_ids = [f"{report_id}_s{j}" for j in range(20)]  # Assume at most 20 sentences
    #     sentence_tensor = self.convert_dict_to_tensor(sentence_features, sentence_ids, device)

    #     if sentence_tensor is not None:
    #         # Project features into the shared space
    #         projected_visual = self.mid_visual_projection(mid_visual_features) # (49, 512)
    #         projected_text = self.sentence_projection(sentence_tensor) #    (49, 512)

    #         # Compute distance
    #         mid_sentence_loss = self.calculate_ot_distance(projected_visual, projected_text)
    #     else:
    #         mid_sentence_loss = torch.tensor(0.0, device=device)

    #     return mid_sentence_loss
    
    # def compute_high_level_alignment(self, high_visual_features, paragraph_features, report_id):
    #     """Compute high-level alignment between visual high-level features and paragraph features."""
    #     device = high_visual_features.device

    #     # Retrieve the paragraph feature
    #     if report_id in paragraph_features:
    #         # Convert paragraph feature to tensor
    #         if isinstance(paragraph_features[report_id], (list, np.ndarray)):
    #             paragraph_tensor = torch.tensor(paragraph_features[report_id], dtype=torch.float).to(device).unsqueeze(0)
    #         else:
    #             paragraph_tensor = paragraph_features[report_id].to(device).unsqueeze(0)

    #         # Repeat paragraph feature to match the number of visual features
    #         paragraph_tensor = paragraph_tensor.repeat(high_visual_features.size(0), 1)

    #         # Project features into the shared space
    #         projected_visual = self.high_visual_projection(high_visual_features)
    #         projected_text = self.paragraph_projection(paragraph_tensor)

    #         # Compute distance
    #         high_paragraph_loss = self.calculate_ot_distance(projected_visual, projected_text)
    #     else:
    #         high_paragraph_loss = torch.tensor(0.0, device=device)

    #     return high_paragraph_loss
    
    # 
    def forward(self, visual_features, text_features, report_ids):
        """
        Forward pass to compute multi-level feature alignment loss.

        Args:
            visual_features: visual feature dict extracted by VisualFeatureExtractor,
                containing 'low_level_feats', 'mid_level_feats', 'high_level_feats',
                'highest_level_feats', 'fused_feats'
            text_features: tuple of text features (paragraph_features, sentence_features, word_features),
                each element is a dict mapping IDs to numpy arrays or tensors
            report_ids: list of report IDs in the current batch

        Returns:
            torch.Tensor: total alignment loss
            dict: per-level loss values for logging
        """
        # Unpack text features
        paragraph_features, sentence_features, word_features = text_features

        # Check if any text feature dict is empty
        if not paragraph_features or not sentence_features or not word_features:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device), {
                'low_word_loss': 0.0,
                'mid_sentence_loss': 0.0,
                'high_paragraph_loss': 0.0,
                'total_loss': 0.0
            }

        # Prepare ID-related lists for the batch
        batch_size = len(report_ids)
        all_losses = {
            'low_word_loss': 0.0,
            'mid_sentence_loss': 0.0,
            'high_paragraph_loss': 0.0,
            'total_loss': 0.0
        }

        total_loss = 0

        # Compute alignment loss for each report in the batch
        for i, report_id in enumerate(report_ids):
            try:
                # Check that the current index is within bounds of the visual features
                if i >= visual_features['low_level_feats'].size(0):
                    print(f"Warning: index {i} exceeds visual feature range {visual_features['low_level_feats'].size(0)}")
                    continue

                # 1. Low-level alignment: visual low-level features vs. word-level features
                if 'low_level_feats' in visual_features:
                    low_visual_features = visual_features['low_level_feats'][i]  # (49, 2048)
                    low_word_loss = self.compute_low_level_alignment(
                        low_visual_features, word_features, report_id)
                else:
                    device = next(self.parameters()).device
                    low_word_loss = torch.tensor(0.0, device=device)

                # 2. Mid-level alignment: visual mid-level features vs. sentence-level features
                if 'mid_level_feats' in visual_features:
                    mid_visual_features = visual_features['mid_level_feats'][i]  # (49, 2048)
                    mid_sentence_loss = self.compute_mid_level_alignment(
                        mid_visual_features, sentence_features, report_id)
                else:
                    device = next(self.parameters()).device
                    mid_sentence_loss = torch.tensor(0.0, device=device)

                # 3. High-level alignment: visual high-level features vs. paragraph-level features
                if 'highest_level_feats' in visual_features:
                    high_visual_features = visual_features['highest_level_feats'][i]  # (49, 2048)
                    high_paragraph_loss = self.compute_high_level_alignment(
                        high_visual_features, paragraph_features, report_id)
                else:
                    device = next(self.parameters()).device
                    high_paragraph_loss = torch.tensor(0.0, device=device)

                # Compute weighted total loss
                report_loss = (self.alpha * low_word_loss +
                              self.beta * mid_sentence_loss +
                              self.gamma * high_paragraph_loss)

                total_loss += report_loss

                # Accumulate into the loss dict
                all_losses['low_word_loss'] += low_word_loss.item()
                all_losses['mid_sentence_loss'] += mid_sentence_loss.item()
                all_losses['high_paragraph_loss'] += high_paragraph_loss.item()

            except Exception as e:
                print(f"Error computing alignment loss for report {report_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Compute average loss
        if batch_size > 0:
            total_loss = total_loss / batch_size
            all_losses['low_word_loss'] /= batch_size
            all_losses['mid_sentence_loss'] /= batch_size
            all_losses['high_paragraph_loss'] /= batch_size
            all_losses['total_loss'] = total_loss.item()

        return total_loss, all_losses

