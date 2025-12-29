"""双塔模型 GPU 优化版本 - 使用 PyTorch"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import SAVE_PATH


class InteractionDataset(Dataset):
    """用户-物品交互数据集"""
    def __init__(self, interactions, user_features, item_features):
        self.interactions = interactions
        self.user_features = torch.FloatTensor(user_features)
        self.item_features = torch.FloatTensor(item_features)
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_idx, item_idx, label = self.interactions[idx]
        return (
            self.user_features[int(user_idx)],
            self.item_features[int(item_idx)],
            torch.FloatTensor([label])
        )


class UserTower(nn.Module):
    """用户塔"""
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2归一化


class ItemTower(nn.Module):
    """物品塔"""
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)  # L2归一化


class TwoTowerModelGPU(nn.Module):
    """双塔模型 GPU版本"""
    def __init__(self, user_dim, item_dim, embedding_dim=64):
        super().__init__()
        self.user_tower = UserTower(user_dim, embedding_dim)
        self.item_tower = ItemTower(item_dim, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, user_feat, item_feat):
        user_emb = self.user_tower(user_feat)
        item_emb = self.item_tower(item_feat)
        # 余弦相似度（已归一化，直接点积）
        score = torch.sum(user_emb * item_emb, dim=1)
        return score
    
    def get_user_embedding(self, user_feat):
        """获取用户向量"""
        with torch.no_grad():
            return self.user_tower(user_feat)
    
    def get_item_embedding(self, item_feat):
        """获取物品向量"""
        with torch.no_grad():
            return self.item_tower(item_feat)


def train_two_tower_gpu(
    all_click_df: pd.DataFrame,
    user_features: np.ndarray,
    item_features: np.ndarray,
    encoders: Dict,
    epochs: int = 5,
    batch_size: int = 1024,
    device: str = 'cuda'
) -> TwoTowerModelGPU:
    """
    训练双塔模型（GPU优化版）
    
    Args:
        all_click_df: 点击数据
        user_features: 用户特征矩阵
        item_features: 物品特征矩阵
        encoders: 特征编码器
        epochs: 训练轮数
        batch_size: 批大小
        device: 设备（cuda/cpu）
    """
    print(f"使用设备: {device}")
    
    # 构建训练数据（正样本）
    click_data = encoders['click_with_features']
    pos_interactions = click_data[['user_idx', 'item_idx']].values
    pos_labels = np.ones(len(pos_interactions))
    
    # 负采样（内存优化：只采样等量负样本）
    n_users = len(user_features)
    n_items = len(item_features)
    n_neg = min(len(pos_interactions), 50000)  # 限制负样本数量
    
    print(f"正样本: {len(pos_interactions)}, 负样本: {n_neg}")
    
    neg_users = np.random.randint(0, n_users, n_neg)
    neg_items = np.random.randint(0, n_items, n_neg)
    neg_interactions = np.column_stack([neg_users, neg_items])
    neg_labels = np.zeros(n_neg)
    
    # 合并正负样本
    all_interactions = np.vstack([
        np.column_stack([pos_interactions, pos_labels]),
        np.column_stack([neg_interactions, neg_labels])
    ])
    np.random.shuffle(all_interactions)
    
    # 释放内存
    del pos_interactions, neg_interactions, pos_labels, neg_labels
    gc.collect()
    
    # 创建数据集和加载器
    dataset = InteractionDataset(all_interactions, user_features, item_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 初始化模型
    model = TwoTowerModelGPU(
        user_dim=user_features.shape[1],
        item_dim=item_features.shape[1],
        embedding_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练
    model.train()
    print(f"\n开始训练双塔模型 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for user_feat, item_feat, labels in pbar:
            user_feat = user_feat.to(device)
            item_feat = item_feat.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            scores = model(user_feat, item_feat)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}")
    
    print("双塔模型训练完成")
    
    # 保存模型
    torch.save(model.state_dict(), SAVE_PATH / 'two_tower_gpu.pth')
    
    # 预计算物品向量（批量处理避免OOM）
    model.eval()
    item_embeddings = []
    item_tensor = torch.FloatTensor(item_features).to(device)
    
    with torch.no_grad():
        for i in range(0, len(item_tensor), batch_size):
            batch = item_tensor[i:i+batch_size]
            emb = model.get_item_embedding(batch).cpu().numpy()
            item_embeddings.append(emb)
    
    item_embeddings = np.vstack(item_embeddings)
    
    # 保存物品向量
    with open(SAVE_PATH / 'item_embeddings_gpu.pkl', 'wb') as f:
        pickle.dump(item_embeddings, f)
    
    print(f"物品向量已保存，形状: {item_embeddings.shape}")
    
    return model


def two_tower_recall_gpu(
    user_id: int,
    user_features_dict: Dict[int, np.ndarray],
    item_embeddings: np.ndarray,
    item_encoder,
    model: TwoTowerModelGPU,
    user_hist_items: set,
    recall_num: int = 10,
    device: str = 'cuda'
) -> List[Tuple[int, float]]:
    """
    使用双塔模型进行召回（GPU优化）
    """
    if user_id not in user_features_dict:
        return []
    
    model.eval()
    with torch.no_grad():
        # 用户特征
        user_feat = torch.FloatTensor(user_features_dict[user_id]).unsqueeze(0).to(device)
        user_emb = model.get_user_embedding(user_feat).cpu().numpy()[0]
        
        # 与所有物品计算相似度（向量化）
        scores = np.dot(item_embeddings, user_emb)
        
        # 排序并过滤历史
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in top_indices:
            item_id = item_encoder.inverse_transform([idx])[0]
            if item_id not in user_hist_items:
                results.append((item_id, float(scores[idx])))
                if len(results) >= recall_num:
                    break
        
        return results


def load_two_tower_model_gpu(device: str = 'cuda') -> Optional[TwoTowerModelGPU]:
    """加载双塔模型（GPU版本）"""
    model_path = SAVE_PATH / 'two_tower_gpu.pth'
    if not model_path.exists():
        return None
    
    # 需要知道模型结构，从encoders获取
    try:
        from .features import load_feature_encoders
        encoders = load_feature_encoders()
        
        # 推断维度
        user_dim = len(encoders['user_feat_cols'])
        item_dim = len(encoders['item_feat_cols'])
        
        model = TwoTowerModelGPU(user_dim, item_dim, embedding_dim=64)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"加载双塔模型失败: {e}")
        return None

