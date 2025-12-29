"""特征工程模块 - 为双塔模型提取用户和物品特征"""
import pickle
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .config import SAVE_PATH


def extract_user_features(all_click_df: pd.DataFrame) -> pd.DataFrame:
    """
    提取用户特征：
    - 点击次数
    - 活跃天数
    - 平均点击时间间隔
    - 点击文章数量
    """
    user_feats = []
    
    for user_id, group in all_click_df.groupby('user_id'):
        click_count = len(group)
        unique_articles = group['click_article_id'].nunique()
        
        # 时间特征
        timestamps = sorted(group['click_timestamp'].values)
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            avg_time_diff = np.mean(time_diffs)
            std_time_diff = np.std(time_diffs)
        else:
            avg_time_diff = 0
            std_time_diff = 0
        
        # 活跃时间范围
        time_span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        user_feats.append({
            'user_id': user_id,
            'click_count': click_count,
            'unique_articles': unique_articles,
            'avg_time_diff': avg_time_diff,
            'std_time_diff': std_time_diff,
            'time_span': time_span,
            'click_rate': click_count / (time_span + 1)  # 避免除0
        })
    
    user_df = pd.DataFrame(user_feats)
    return user_df


def extract_item_features(all_click_df: pd.DataFrame) -> pd.DataFrame:
    """
    提取物品（文章）特征：
    - 被点击次数
    - 被多少用户点击
    - 平均点击时间
    - 时间热度（最近被点击的权重更高）
    """
    item_feats = []
    
    for item_id, group in all_click_df.groupby('click_article_id'):
        click_count = len(group)
        unique_users = group['user_id'].nunique()
        
        # 时间特征
        timestamps = group['click_timestamp'].values
        avg_timestamp = np.mean(timestamps)
        
        # 时间热度（指数衰减）
        max_time = all_click_df['click_timestamp'].max()
        time_weights = np.exp(-(max_time - timestamps) / (86400 * 7))  # 7天衰减
        time_popularity = np.sum(time_weights)
        
        item_feats.append({
            'click_article_id': item_id,
            'click_count': click_count,
            'unique_users': unique_users,
            'avg_timestamp': avg_timestamp,
            'time_popularity': time_popularity,
            'click_per_user': click_count / unique_users
        })
    
    item_df = pd.DataFrame(item_feats)
    return item_df


def build_feature_matrix(
    all_click_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    构建用户-物品特征矩阵用于双塔模型训练
    返回：(user_features, item_features, encoders)
    """
    # 合并特征
    click_with_features = all_click_df.merge(user_df, on='user_id', how='left')
    click_with_features = click_with_features.merge(item_df, on='click_article_id', how='left')
    
    # 用户ID编码
    user_encoder = LabelEncoder()
    click_with_features['user_idx'] = user_encoder.fit_transform(click_with_features['user_id'])
    
    # 物品ID编码
    item_encoder = LabelEncoder()
    click_with_features['item_idx'] = item_encoder.fit_transform(click_with_features['click_article_id'])
    
    # 用户特征列
    user_feat_cols = ['click_count', 'unique_articles', 'avg_time_diff', 
                      'std_time_diff', 'time_span', 'click_rate']
    
    # 物品特征列
    item_feat_cols = ['click_count', 'unique_users', 'avg_timestamp', 
                      'time_popularity', 'click_per_user']
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    user_scaler = StandardScaler()
    item_scaler = StandardScaler()
    
    user_features_scaled = user_scaler.fit_transform(user_df[user_feat_cols].fillna(0))
    item_features_scaled = item_scaler.fit_transform(item_df[item_feat_cols].fillna(0))
    
    encoders = {
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'user_scaler': user_scaler,
        'item_scaler': item_scaler,
        'user_feat_cols': user_feat_cols,
        'item_feat_cols': item_feat_cols,
        'click_with_features': click_with_features
    }
    
    # 保存编码器
    with open(SAVE_PATH / 'feature_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    return user_features_scaled, item_features_scaled, encoders


def load_feature_encoders():
    """加载特征编码器"""
    with open(SAVE_PATH / 'feature_encoders.pkl', 'rb') as f:
        return pickle.load(f)

