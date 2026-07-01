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
        计算两组特征之间的最优传输
        
        Args:
            weight1 : (N, D) 视觉特征
            weight2 : (M, D) 文本特征
        
        Return:
            flow : (N, M) 流矩阵
            dist : (1, ) 距离值
        """
        # 确保输入是2D张量
        if weight1.dim() == 1:
            weight1 = weight1.unsqueeze(0)  # 添加批次维度
        if weight2.dim() == 1:
            weight2 = weight2.unsqueeze(0)  # 添加批次维度
            
        # 如果特征为空（某个维度为0），则返回零距离
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
            flow = flow.type(torch.FloatTensor).to(weight1.device)  # 修改为使用相同的设备
            
            dist = self.cost_map * flow  # (N, M)
            dist = torch.sum(dist)  # (1,) float
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
    多级别特征对齐模块，包含可学习的投影层
    """
    def __init__(self, visual_dims=(2048, 2048, 2048), 
                 text_dims=(768, 768, 768),
                 projection_dim=512,
                 alpha=0.4, beta=0.2, gamma=0.4,
                 ot_impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5):
        """
        初始化多级别特征对齐模块
        
        Args:
            visual_dims: 视觉特征的维度元组 (低级维度, 中级维度, 高级维度)
            text_dims: 文本特征的维度元组 (单词维度, 句子维度, 段落维度)
            projection_dim: 投影后的共同维度
            alpha: 低级别特征(visual-word)对齐损失权重
            beta: 中级别特征(visual-sentence)对齐损失权重
            gamma: 高级别特征(visual-paragraph)对齐损失权重
            ot_impl: OT实现方法
            ot_reg: OT正则化参数
            ot_tau: UOT参数
        """
        super().__init__()
        
        self.projection_dim = projection_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # 创建OT计算器
        self.ot_calculator = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)
        
        # 创建投影层 - 低级别 (word-level)
        self.low_visual_projection = nn.Linear(visual_dims[0], projection_dim)
        self.word_projection = nn.Linear(text_dims[0], projection_dim)
        
        # 创建投影层 - 中级别 (sentence-level)
        self.mid_visual_projection = nn.Linear(visual_dims[1], projection_dim)
        self.sentence_projection = nn.Linear(text_dims[1], projection_dim)
        
        # 创建投影层 - 高级别 (paragraph-level)
        self.high_visual_projection = nn.Linear(visual_dims[2], projection_dim)
        self.paragraph_projection = nn.Linear(text_dims[2], projection_dim)
        
        # 初始化投影层权重
        self._init_projection_weights()
    
    def _init_projection_weights(self):
        """初始化投影层权重"""
        for module in [self.low_visual_projection, self.word_projection,
                       self.mid_visual_projection, self.sentence_projection,
                       self.high_visual_projection, self.paragraph_projection]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    

    def convert_dict_to_tensor(self, feature_dict, ids_list, device):
        """将特征字典转换为tensor"""
        features = []
        valid_ids = []
        
        # 首先收集所有有效的特征
        for id_key in ids_list:
            if id_key in feature_dict:
                # 如果是numpy数组或列表，转换为tensor
                if isinstance(feature_dict[id_key], (list, np.ndarray)):
                    # 检查特征是否为空
                    if len(feature_dict[id_key]) > 0:
                        features.append(torch.tensor(feature_dict[id_key], dtype=torch.float).to(device))
                        valid_ids.append(id_key)
                # 如果已经是tensor，确保在正确设备上
                elif isinstance(feature_dict[id_key], torch.Tensor):
                    # 检查特征是否为空
                    if feature_dict[id_key].numel() > 0:
                        features.append(feature_dict[id_key].to(device))
                        valid_ids.append(id_key)
        
        # 如果没有收集到任何有效特征，返回None
        if not features:
            return None
            
        # 检查所有特征的维度是否相同
        feat_dim = features[0].size(-1)
        for i, feat in enumerate(features):
            if feat.size(-1) != feat_dim:
                print(f"警告: 特征维度不匹配! ID {valid_ids[i]} 的维度是 {feat.size(-1)}, 应该是 {feat_dim}")
                # 调整维度 (截断或填充)
                if feat.size(-1) > feat_dim:
                    features[i] = feat[..., :feat_dim]
                else:
                    pad_size = feat_dim - feat.size(-1)
                    features[i] = torch.cat([feat, torch.zeros((*feat.shape[:-1], pad_size), device=device)], dim=-1)
        
        # 尝试堆叠特征，如果维度不同，则进行处理
        try:
            return torch.stack(features)
        except RuntimeError as e:
            print(f"堆叠特征时出错: {e}")
            print(f"特征形状: {[f.shape for f in features]}")
            
            # 尝试调整所有特征为相同形状
            max_dims = [max(f.size(i) for f in features) for i in range(features[0].dim())]
            adjusted_features = []
            
            for feat in features:
                if feat.dim() != len(max_dims):
                    print(f"警告: 特征的维度数量不同 {feat.dim()} vs {len(max_dims)}")
                    continue
                    
                # 调整每个维度
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
    #     """将特征字典转换为tensor"""
    #     features = []
    #     for id_key in ids_list:
    #         if id_key in feature_dict:
    #             # 如果是numpy数组，转换为tensor
    #             if isinstance(feature_dict[id_key], (list, np.ndarray)):
    #                 features.append(torch.tensor(feature_dict[id_key], dtype=torch.float).to(device))
    #             # 如果已经是tensor，确保在正确设备上
    #             elif isinstance(feature_dict[id_key], torch.Tensor):
    #                 features.append(feature_dict[id_key].to(device))
        
    #     if features:
    #         return torch.stack(features)
    #     else:
    #         return None
    
    # 
    def calculate_ot_distance(self, visual_features, text_features):
        """计算投影后的特征之间的OT距离"""
        # 检查特征是否为空
        if text_features is None or visual_features.size(0) == 0:
            return torch.tensor(0.0, device=visual_features.device)
            
        # 检查文本特征的维度和大小
        if text_features.dim() == 0 or text_features.size(0) == 0:
            return torch.tensor(0.0, device=visual_features.device)
        
        # 确保特征具有正确的维度
        if visual_features.dim() == 1:
            visual_features = visual_features.unsqueeze(0)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        
        try:
            _, dist = self.ot_calculator(visual_features.unsqueeze(1), text_features.unsqueeze(1))
            # 归一化距离
            normalized_dist = dist / (visual_features.size(0) * text_features.size(0))
            return normalized_dist
        except RuntimeError as e:
            print(f"计算OT距离时出错: {e}")
            print(f"视觉特征形状: {visual_features.shape}, 文本特征形状: {text_features.shape}")
            return torch.tensor(0.0, device=visual_features.device)
    
    def compute_low_level_alignment(self, low_visual_features, word_features, report_id):
        """计算低级别(视觉低级特征与单词特征)对齐"""
        device = low_visual_features.device
        
        # 获取与报告相关的单词ID
        word_ids = [f"{report_id}_w{j}" for j in range(100)]  # 假设最多100个单词
        word_tensor = self.convert_dict_to_tensor(word_features, word_ids, device)
        
        if word_tensor is not None and word_tensor.size(0) > 0:
            # 投影特征到共同空间
            projected_visual = self.low_visual_projection(low_visual_features)
            projected_text = self.word_projection(word_tensor)
            
            # 计算距离
            low_word_loss = self.calculate_ot_distance(projected_visual, projected_text)
        else:
            low_word_loss = torch.tensor(0.0, device=device)
        
        return low_word_loss
    
    def compute_mid_level_alignment(self, mid_visual_features, sentence_features, report_id):
        """计算中级别(视觉中级特征与句子特征)对齐"""
        device = mid_visual_features.device
        
        # 获取与报告相关的句子ID
        sentence_ids = [f"{report_id}_s{j}" for j in range(20)]  # 假设最多20个句子
        sentence_tensor = self.convert_dict_to_tensor(sentence_features, sentence_ids, device)
        
        if sentence_tensor is not None and sentence_tensor.size(0) > 0:
            # 投影特征到共同空间
            projected_visual = self.mid_visual_projection(mid_visual_features)
            projected_text = self.sentence_projection(sentence_tensor)
            
            # 计算距离
            mid_sentence_loss = self.calculate_ot_distance(projected_visual, projected_text)
        else:
            mid_sentence_loss = torch.tensor(0.0, device=device)
        
        return mid_sentence_loss
    
    def compute_high_level_alignment(self, high_visual_features, paragraph_features, report_id):
        """计算高级别(视觉高级特征与段落特征)对齐"""
        device = high_visual_features.device
        
        # 获取段落特征
        if report_id in paragraph_features:
            # 转换段落特征为tensor
            if isinstance(paragraph_features[report_id], (list, np.ndarray)):
                paragraph_tensor = torch.tensor(paragraph_features[report_id], dtype=torch.float).to(device).unsqueeze(0)
            else:
                paragraph_tensor = paragraph_features[report_id].to(device).unsqueeze(0)
            
            # 如果特征是空的，返回零损失
            if paragraph_tensor.numel() == 0:
                return torch.tensor(0.0, device=device)
                
            # 复制段落特征以匹配视觉特征的数量
            paragraph_tensor = paragraph_tensor.repeat(high_visual_features.size(0), 1)
            
            # 投影特征到共同空间
            projected_visual = self.high_visual_projection(high_visual_features)
            projected_text = self.paragraph_projection(paragraph_tensor)
            
            # 计算距离
            high_paragraph_loss = self.calculate_ot_distance(projected_visual, projected_text)
        else:
            high_paragraph_loss = torch.tensor(0.0, device=device)
        
        return high_paragraph_loss
    
    # def compute_low_level_alignment(self, low_visual_features, word_features, report_id):
    #     """计算低级别(视觉低级特征与单词特征)对齐"""
    #     device = low_visual_features.device
        
    #     # 获取与报告相关的单词ID
    #     word_ids = [f"{report_id}_w{j}" for j in range(100)]  # 假设最多100个单词
    #     word_tensor = self.convert_dict_to_tensor(word_features, word_ids, device)
        
    #     if word_tensor is not None:
    #         # 投影特征到共同空间
    #         projected_visual = self.low_visual_projection(low_visual_features)
    #         projected_text = self.word_projection(word_tensor)
            
    #         # 计算距离
    #         low_word_loss = self.calculate_ot_distance(projected_visual, projected_text)
    #     else:
    #         low_word_loss = torch.tensor(0.0, device=device)
        
    #     return low_word_loss
    
    # def compute_mid_level_alignment(self, mid_visual_features, sentence_features, report_id):
    #     """计算中级别(视觉中级特征与句子特征)对齐"""
    #     device = mid_visual_features.device
        
    #     # 获取与报告相关的句子ID
    #     sentence_ids = [f"{report_id}_s{j}" for j in range(20)]  # 假设最多20个句子
    #     sentence_tensor = self.convert_dict_to_tensor(sentence_features, sentence_ids, device)
        
    #     if sentence_tensor is not None:
    #         # 投影特征到共同空间
    #         projected_visual = self.mid_visual_projection(mid_visual_features) # (49, 512)
    #         projected_text = self.sentence_projection(sentence_tensor) #    (49, 512)
            
    #         # 计算距离
    #         mid_sentence_loss = self.calculate_ot_distance(projected_visual, projected_text)
    #     else:
    #         mid_sentence_loss = torch.tensor(0.0, device=device)
        
    #     return mid_sentence_loss
    
    # def compute_high_level_alignment(self, high_visual_features, paragraph_features, report_id):
    #     """计算高级别(视觉高级特征与段落特征)对齐"""
    #     device = high_visual_features.device
        
    #     # 获取段落特征
    #     if report_id in paragraph_features:
    #         # 转换段落特征为tensor
    #         if isinstance(paragraph_features[report_id], (list, np.ndarray)):
    #             paragraph_tensor = torch.tensor(paragraph_features[report_id], dtype=torch.float).to(device).unsqueeze(0)
    #         else:
    #             paragraph_tensor = paragraph_features[report_id].to(device).unsqueeze(0)
            
    #         # 复制段落特征以匹配视觉特征的数量
    #         paragraph_tensor = paragraph_tensor.repeat(high_visual_features.size(0), 1)
            
    #         # 投影特征到共同空间
    #         projected_visual = self.high_visual_projection(high_visual_features)
    #         projected_text = self.paragraph_projection(paragraph_tensor)
            
    #         # 计算距离
    #         high_paragraph_loss = self.calculate_ot_distance(projected_visual, projected_text)
    #     else:
    #         high_paragraph_loss = torch.tensor(0.0, device=device)
        
    #     return high_paragraph_loss
    
    # 
    def forward(self, visual_features, text_features, report_ids):
        """
        前向传播计算多级别特征对齐损失
        
        Args:
            visual_features: 视觉特征字典，由VisualFeatureExtractor提取
                包含 'low_level_feats', 'mid_level_feats', 'high_level_feats', 'highest_level_feats', 'fused_feats'
            text_features: 文本特征元组 (paragraph_features, sentence_features, word_features)
                每个元素都是字典，键为ID，值为numpy数组或tensor
            report_ids: 当前批次的报告ID列表
            
        Returns:
            torch.Tensor: 总的对齐损失
            dict: 包含各级别损失的字典，用于记录
        """
        # 解包文本特征
        paragraph_features, sentence_features, word_features = text_features
        
        # 检查文本特征是否为空
        if not paragraph_features or not sentence_features or not word_features:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device), {
                'low_word_loss': 0.0,
                'mid_sentence_loss': 0.0,
                'high_paragraph_loss': 0.0,
                'total_loss': 0.0
            }
        
        # 准备报告相关的ID列表
        batch_size = len(report_ids)
        all_losses = {
            'low_word_loss': 0.0,
            'mid_sentence_loss': 0.0,
            'high_paragraph_loss': 0.0,
            'total_loss': 0.0
        }
        
        total_loss = 0
        
        # 对批次中的每个报告计算对齐损失
        for i, report_id in enumerate(report_ids):
            try:
                # 检查当前索引是否在视觉特征的范围内
                if i >= visual_features['low_level_feats'].size(0):
                    print(f"警告: 索引 {i} 超出视觉特征的范围 {visual_features['low_level_feats'].size(0)}")
                    continue
                
                # 1. 低级别对齐：visual low-level features 与 word-level features
                if 'low_level_feats' in visual_features:
                    low_visual_features = visual_features['low_level_feats'][i]  # (49, 2048)
                    low_word_loss = self.compute_low_level_alignment(
                        low_visual_features, word_features, report_id)
                else:
                    device = next(self.parameters()).device
                    low_word_loss = torch.tensor(0.0, device=device)
                
                # 2. 中级别对齐：visual mid-level features 与 sentence-level features
                if 'mid_level_feats' in visual_features:
                    mid_visual_features = visual_features['mid_level_feats'][i]  # (49, 2048)
                    mid_sentence_loss = self.compute_mid_level_alignment(
                        mid_visual_features, sentence_features, report_id)
                else:
                    device = next(self.parameters()).device
                    mid_sentence_loss = torch.tensor(0.0, device=device)
                
                # 3. 高级别对齐：visual high-level features 与 paragraph-level features
                if 'highest_level_feats' in visual_features:
                    high_visual_features = visual_features['highest_level_feats'][i]  # (49, 2048)
                    high_paragraph_loss = self.compute_high_level_alignment(
                        high_visual_features, paragraph_features, report_id)
                else:
                    device = next(self.parameters()).device
                    high_paragraph_loss = torch.tensor(0.0, device=device)
                
                # 加权计算总损失
                report_loss = (self.alpha * low_word_loss + 
                              self.beta * mid_sentence_loss + 
                              self.gamma * high_paragraph_loss)
                
                total_loss += report_loss
                
                # 累加到所有损失字典中
                all_losses['low_word_loss'] += low_word_loss.item()
                all_losses['mid_sentence_loss'] += mid_sentence_loss.item()
                all_losses['high_paragraph_loss'] += high_paragraph_loss.item()
            
            except Exception as e:
                print(f"计算报告 {report_id} 的对齐损失时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算平均损失
        if batch_size > 0:
            total_loss = total_loss / batch_size
            all_losses['low_word_loss'] /= batch_size
            all_losses['mid_sentence_loss'] /= batch_size
            all_losses['high_paragraph_loss'] /= batch_size
            all_losses['total_loss'] = total_loss.item()
        
        return total_loss, all_losses

