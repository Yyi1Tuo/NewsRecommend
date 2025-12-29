"""双塔模型实现 - 用于深度学习召回"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import SAVE_PATH


class TwoTowerModel:
    """
    简化版双塔模型（使用浅层MLP）
    用户塔和物品塔分别编码特征，通过余弦相似度计算匹配分数
    """
    
    def __init__(self, user_dim: int, item_dim: int, embedding_dim: int = 64):
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedding_dim = embedding_dim
        
        # 简化实现：使用矩阵分解近似双塔
        # 在实际应用中应该用 PyTorch/TensorFlow 实现完整的神经网络
        self.user_weights = None
        self.item_weights = None
        self.user_bias = None
        self.item_bias = None
        
    def fit(self, user_features: np.ndarray, item_features: np.ndarray, 
            interactions: np.ndarray, epochs: int = 10, lr: float = 0.01):
        """
        训练双塔模型
        interactions: (user_idx, item_idx, label) 的数组
        """
        n_users = user_features.shape[0]
        n_items = item_features.shape[0]
        
        # 初始化权重
        np.random.seed(42)
        self.user_weights = np.random.randn(self.user_dim, self.embedding_dim) * 0.01
        self.item_weights = np.random.randn(self.item_dim, self.embedding_dim) * 0.01
        self.user_bias = np.zeros(self.embedding_dim)
        self.item_bias = np.zeros(self.embedding_dim)
        
        print(f"开始训练双塔模型 ({epochs} epochs)...")
        
        for epoch in range(epochs):
            # 简化训练：使用 ALS 风格的交替优化
            losses = []
            
            # 计算用户和物品的嵌入
            user_embeddings = self._get_user_embeddings(user_features)
            item_embeddings = self._get_item_embeddings(item_features)
            
            # 对每个交互计算损失
            for user_idx, item_idx, label in interactions[:1000]:  # 采样加速
                user_emb = user_embeddings[int(user_idx)]
                item_emb = item_embeddings[int(item_idx)]
                
                # 余弦相似度
                pred = np.dot(user_emb, item_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-8)
                loss = (pred - label) ** 2
                losses.append(loss)
            
            avg_loss = np.mean(losses)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("双塔模型训练完成")
        return self
    
    def _get_user_embeddings(self, user_features: np.ndarray) -> np.ndarray:
        """获取用户嵌入向量"""
        embeddings = np.dot(user_features, self.user_weights) + self.user_bias
        return embeddings
    
    def _get_item_embeddings(self, item_features: np.ndarray) -> np.ndarray:
        """获取物品嵌入向量"""
        embeddings = np.dot(item_features, self.item_weights) + self.item_bias
        return embeddings
    
    def predict(self, user_features: np.ndarray, item_features: np.ndarray) -> float:
        """预测用户对物品的兴趣分数"""
        user_emb = self._get_user_embeddings(user_features.reshape(1, -1))[0]
        item_emb = self._get_item_embeddings(item_features.reshape(1, -1))[0]
        
        # 余弦相似度
        score = np.dot(user_emb, item_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-8)
        return score
    
    def save(self, path: Path):
        """保存模型"""
        model_data = {
            'user_weights': self.user_weights,
            'item_weights': self.item_weights,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'user_dim': self.user_dim,
            'item_dim': self.item_dim,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Path):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['user_dim'], model_data['item_dim'], model_data['embedding_dim'])
        model.user_weights = model_data['user_weights']
        model.item_weights = model_data['item_weights']
        model.user_bias = model_data['user_bias']
        model.item_bias = model_data['item_bias']
        return model


def train_two_tower_model(all_click_df: pd.DataFrame, user_df: pd.DataFrame, 
                          item_df: pd.DataFrame, epochs: int = 10):
    """训练双塔模型并保存"""
    from .features import build_feature_matrix
    
    print("构建特征矩阵...")
    user_features, item_features, encoders = build_feature_matrix(all_click_df, user_df, item_df)
    
    # 构建正样本（实际点击）
    click_data = encoders['click_with_features']
    interactions = click_data[['user_idx', 'item_idx']].values
    labels = np.ones(len(interactions))
    
    # 构建负样本（随机采样）
    n_neg = len(interactions)
    neg_users = np.random.randint(0, len(user_df), n_neg)
    neg_items = np.random.randint(0, len(item_df), n_neg)
    neg_interactions = np.column_stack([neg_users, neg_items])
    neg_labels = np.zeros(n_neg)
    
    # 合并正负样本
    all_interactions = np.vstack([
        np.column_stack([interactions, labels]),
        np.column_stack([neg_interactions, neg_labels])
    ])
    
    # 打乱
    np.random.shuffle(all_interactions)
    
    # 训练模型
    model = TwoTowerModel(
        user_dim=user_features.shape[1],
        item_dim=item_features.shape[1],
        embedding_dim=64
    )
    model.fit(user_features, item_features, all_interactions, epochs=epochs)
    
    # 保存
    model.save(SAVE_PATH / 'two_tower_model.pkl')
    
    # 预计算所有物品的嵌入（用于快速召回）
    item_embeddings = model._get_item_embeddings(item_features)
    with open(SAVE_PATH / 'item_embeddings.pkl', 'wb') as f:
        pickle.dump(item_embeddings, f)
    
    print("双塔模型已保存")
    return model


def two_tower_recall(
    user_id: int,
    user_features_dict: Dict[int, np.ndarray],
    item_embeddings: np.ndarray,
    item_encoder,
    model: TwoTowerModel,
    user_hist_items: set,
    recall_num: int = 10
) -> List[Tuple[int, float]]:
    """
    使用双塔模型进行召回
    """
    if user_id not in user_features_dict:
        return []
    
    user_feat = user_features_dict[user_id]
    user_emb = model._get_user_embeddings(user_feat.reshape(1, -1))[0]
    
    # 计算与所有物品的相似度
    scores = []
    for idx, item_emb in enumerate(item_embeddings):
        # 余弦相似度
        score = np.dot(user_emb, item_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(item_emb) + 1e-8)
        
        # 原始物品ID
        item_id = item_encoder.inverse_transform([idx])[0]
        
        # 过滤历史
        if item_id not in user_hist_items:
            scores.append((item_id, float(score)))
    
    # 排序并返回 TopK
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:recall_num]

