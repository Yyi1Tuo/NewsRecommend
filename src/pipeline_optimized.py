"""内存优化版 Pipeline - 支持分批处理和GPU加速"""
import pickle
import gc
import psutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import (
    DATA_PATH,
    SAVE_PATH,
    I2I_SIM_FILENAME,
    SIM_ITEM_TOPK,
    RECALL_ITEM_NUM,
    ITEM_TOPK_K,
    MODEL_NAME,
)
from . import data as data_mod
from . import similarity
from . import recall
from . import submit as submit_mod


def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_memory_stats(stage=""):
    """打印内存统计"""
    mem_used = get_memory_usage()
    mem_available = psutil.virtual_memory().available / 1024 / 1024
    print(f"[{stage}] 内存使用: {mem_used:.1f}MB, 可用: {mem_available:.1f}MB")


def run_lightweight(
    topk_submit: int = 5,
    use_gpu: bool = True,
    batch_size: int = 1000
) -> Path:
    """
    轻量级运行模式 - 内存优化版
    
    Args:
        topk_submit: 提交的 TopK
        use_gpu: 是否使用GPU（如果可用）
        batch_size: 每批处理的用户数
    """
    print("=" * 70)
    print("轻量级推荐系统 - 内存优化版")
    print("=" * 70)
    
    # 检测GPU
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"计算设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print_memory_stats("启动")
    
    # 1. 加载数据（只读必要字段）
    print("\n[1/5] 加载数据...")
    all_click_df = data_mod.get_all_click_df(offline=False)
    print(f"点击数据: {len(all_click_df)} 条")
    print_memory_stats("数据加载")
    
    # 2. 基础统计
    print("\n[2/5] 计算基础统计...")
    user_item_time_dict = data_mod.get_user_item_time(all_click_df)
    item_topk_click = data_mod.get_item_topk_click(all_click_df, k=ITEM_TOPK_K).tolist()
    
    # 获取所有用户
    all_users = all_click_df["user_id"].unique()
    print(f"用户数: {len(all_users)}")
    print_memory_stats("统计完成")
    
    # 3. ItemCF 相似度（如果不存在才计算）
    print("\n[3/5] 准备ItemCF相似度...")
    i2i_sim = _load_or_build_i2i(all_click_df)
    print_memory_stats("相似度加载")
    
    # 释放不需要的数据
    del all_click_df
    gc.collect()
    print_memory_stats("释放数据")
    
    # 4. 分批召回
    print(f"\n[4/5] 分批召回（批大小: {batch_size}）...")
    user_recall_dict = {}
    
    n_batches = (len(all_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_users))
        batch_users = all_users[start_idx:end_idx]
        
        print(f"  批次 {batch_idx+1}/{n_batches}: 处理用户 {start_idx}-{end_idx}")
        
        for user in tqdm(batch_users, desc=f"召回批次{batch_idx+1}"):
            recs = recall.item_based_recommend(
                user_id=user,
                user_item_time_dict=user_item_time_dict,
                i2i_sim=i2i_sim,
                sim_item_topk=SIM_ITEM_TOPK,
                recall_item_num=RECALL_ITEM_NUM,
                item_topk_click=item_topk_click,
            )
            user_recall_dict[user] = recs
        
        # 定期清理内存
        if (batch_idx + 1) % 5 == 0:
            gc.collect()
            print_memory_stats(f"批次{batch_idx+1}完成")
    
    print(f"召回完成，共 {len(user_recall_dict)} 个用户")
    print_memory_stats("召回完成")
    
    # 5. 转换为DataFrame（流式处理）
    print("\n[5/5] 生成提交文件...")
    
    # 只保留测试集用户
    tst_click = pd.read_csv(str(DATA_PATH / "testA_click_log.csv"))
    tst_users = set(tst_click["user_id"].unique())
    
    # 流式构建DataFrame
    user_item_score_list = []
    for user in tqdm(tst_users, desc="构建结果"):
        if user in user_recall_dict:
            for item, score in user_recall_dict[user]:
                user_item_score_list.append([user, item, score])
    
    recall_df = pd.DataFrame(
        user_item_score_list, 
        columns=["user_id", "click_article_id", "pred_score"]
    )
    
    print(f"测试集召回: {len(recall_df)} 条")
    print_memory_stats("结果构建")
    
    # 生成提交
    submit_path = submit_mod.submit(recall_df, topk=topk_submit, model_name=f"{MODEL_NAME}_lite")
    
    print(f"\n提交文件: {submit_path}")
    print_memory_stats("完成")
    print("=" * 70)
    
    return submit_path


def run_multi_gpu(
    topk_submit: int = 5,
    use_itemcf: bool = True,
    use_two_tower: bool = True,
    train_two_tower: bool = False,
    batch_size: int = 500,
    device: str = 'cuda'
) -> Path:
    """
    多路召回 GPU优化版
    
    Args:
        topk_submit: 提交TopK
        use_itemcf: 是否使用ItemCF
        use_two_tower: 是否使用双塔模型
        train_two_tower: 是否训练双塔
        batch_size: 批大小
        device: 计算设备
    """
    print("=" * 70)
    print("多路召回系统 - GPU优化版")
    print("=" * 70)
    
    # 检测GPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        device = 'cpu'
    
    print(f"计算设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print_memory_stats("启动")
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    all_click_df = data_mod.get_all_click_df(offline=False)
    user_item_time_dict = data_mod.get_user_item_time(all_click_df)
    item_topk_click = data_mod.get_item_topk_click(all_click_df, k=ITEM_TOPK_K).tolist()
    all_users = all_click_df["user_id"].unique()
    
    print(f"用户: {len(all_users)}, 点击: {len(all_click_df)}")
    print_memory_stats("数据加载")
    
    # 2. 准备召回资源
    print("\n[2/5] 准备召回资源...")
    
    i2i_sim = None
    if use_itemcf:
        i2i_sim = _load_or_build_i2i(all_click_df)
        print("ItemCF 相似度已加载")
    
    two_tower_model = None
    item_embeddings = None
    user_features_dict = None
    item_encoder = None
    
    if use_two_tower:
        if train_two_tower or not (SAVE_PATH / 'two_tower_gpu.pth').exists():
            print("\n训练双塔模型...")
            from . import features as feat_mod
            from . import two_tower_gpu
            
            # 提取特征
            user_df = feat_mod.extract_user_features(all_click_df)
            item_df = feat_mod.extract_item_features(all_click_df)
            user_features, item_features, encoders = feat_mod.build_feature_matrix(
                all_click_df, user_df, item_df
            )
            
            # 训练
            two_tower_model = two_tower_gpu.train_two_tower_gpu(
                all_click_df,
                user_features,
                item_features,
                encoders,
                epochs=3,  # 减少epoch
                batch_size=2048,  # 增大batch减少迭代
                device=device
            )
            
            # 清理
            del user_df, item_df, user_features, item_features
            gc.collect()
        
        # 加载模型和向量
        if (SAVE_PATH / 'two_tower_gpu.pth').exists():
            print("加载双塔模型...")
            from . import two_tower_gpu
            from . import features as feat_mod
            
            two_tower_model = two_tower_gpu.load_two_tower_model_gpu(device)
            
            with open(SAVE_PATH / 'item_embeddings_gpu.pkl', 'rb') as f:
                item_embeddings = pickle.load(f)
            
            encoders = feat_mod.load_feature_encoders()
            item_encoder = encoders['item_encoder']
            
            # 构建用户特征字典（轻量级）
            user_df = feat_mod.extract_user_features(all_click_df)
            user_features, _, _ = feat_mod.build_feature_matrix(
                all_click_df, user_df, 
                feat_mod.extract_item_features(all_click_df)
            )
            user_encoder = encoders['user_encoder']
            user_features_dict = {
                uid: user_features[user_encoder.transform([uid])[0]]
                for uid in all_users
            }
            
            del user_df, user_features
            gc.collect()
            
            print("双塔模型已加载")
    
    print_memory_stats("资源准备")
    
    # 释放原始数据
    del all_click_df
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    print_memory_stats("释放数据")
    
    # 3. 分批召回
    print(f"\n[3/5] 分批多路召回（批大小: {batch_size}）...")
    user_recall_dict = {}
    
    n_batches = (len(all_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_users))
        batch_users = all_users[start_idx:end_idx]
        
        print(f"  批次 {batch_idx+1}/{n_batches}: {start_idx}-{end_idx}")
        
        # 统计信息
        batch_stats = {
            'itemcf_success': 0,
            'itemcf_fail': 0,
            'two_tower_success': 0,
            'two_tower_fail': 0,
            'itemcf_items': [],
            'two_tower_items': []
        }
        
        for user_idx, user in enumerate(tqdm(batch_users, desc=f"召回{batch_idx+1}")):
            recall_results = {}
            
            # ItemCF召回
            if use_itemcf and i2i_sim:
                try:
                    itemcf_recs = recall.item_based_recommend(
                        user_id=user,
                        user_item_time_dict=user_item_time_dict,
                        i2i_sim=i2i_sim,
                        sim_item_topk=SIM_ITEM_TOPK,
                        recall_item_num=RECALL_ITEM_NUM,
                        item_topk_click=item_topk_click,
                    )
                    recall_results['itemcf'] = itemcf_recs
                    batch_stats['itemcf_success'] += 1
                    if itemcf_recs:
                        batch_stats['itemcf_items'].append(len(itemcf_recs))
                except Exception as e:
                    recall_results['itemcf'] = []
                    batch_stats['itemcf_fail'] += 1
                    if user_idx == 0:  # 只打印第一个用户的错误
                        print(f"\n  ItemCF召回失败: {e}")
            
            # 双塔召回
            if use_two_tower and two_tower_model:
                try:
                    from . import two_tower_gpu
                    user_hist_items = {item_id for item_id, _ in user_item_time_dict.get(user, [])}
                    tt_recs = two_tower_gpu.two_tower_recall_gpu(
                        user_id=user,
                        user_features_dict=user_features_dict,
                        item_embeddings=item_embeddings,
                        item_encoder=item_encoder,
                        model=two_tower_model,
                        user_hist_items=user_hist_items,
                        recall_num=RECALL_ITEM_NUM,
                        device=device
                    )
                    recall_results['two_tower'] = tt_recs
                    batch_stats['two_tower_success'] += 1
                    if tt_recs:
                        batch_stats['two_tower_items'].append(len(tt_recs))
                    
                    # 调试：打印第一个用户的召回结果
                    if user_idx == 0 and batch_idx == 0:
                        print(f"\n  [调试] 用户{user}:")
                        print(f"    ItemCF召回: {len(itemcf_recs) if 'itemcf' in recall_results else 0} 个")
                        if itemcf_recs:
                            print(f"      示例: {itemcf_recs[:3]}")
                        print(f"    双塔召回: {len(tt_recs)} 个")
                        if tt_recs:
                            print(f"      示例: {tt_recs[:3]}")
                        
                except Exception as e:
                    recall_results['two_tower'] = []
                    batch_stats['two_tower_fail'] += 1
                    if user_idx == 0:  # 只打印第一个用户的错误
                        print(f"\n  双塔召回失败: {e}")
                        import traceback
                        traceback.print_exc()
            
            # 融合：归一化分数后加权
            item_scores = defaultdict(float)
            
            for strategy, items in recall_results.items():
                if not items:
                    continue
                
                # 归一化分数到 [0, 1]
                scores = [s for _, s in items]
                if len(scores) > 0:
                    min_score = min(scores)
                    max_score = max(scores)
                    if max_score > min_score:
                        norm_items = [
                            (item, (score - min_score) / (max_score - min_score))
                            for item, score in items
                        ]
                    else:
                        norm_items = [(item, 1.0) for item, _ in items]
                else:
                    norm_items = items
                
                # 加权
                weight = 0.5 if strategy == 'itemcf' else 0.5  # 平等权重
                for item, norm_score in norm_items:
                    item_scores[item] += weight * norm_score
            
            # 排序并取TopK
            if item_scores:
                final_recs = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:RECALL_ITEM_NUM * 2]
            else:
                # 如果没有召回，使用热门
                final_recs = [(item, 1.0 - idx*0.01) for idx, item in enumerate(item_topk_click[:RECALL_ITEM_NUM])]
            
            user_recall_dict[user] = final_recs
            
            # 调试：打印第一个用户的融合结果
            if user_idx == 0 and batch_idx == 0:
                print(f"    融合后: {len(final_recs)} 个")
                print(f"      示例: {final_recs[:5]}")
        
        # 打印批次统计
        print(f"\n  批次{batch_idx+1}统计:")
        print(f"    ItemCF: 成功={batch_stats['itemcf_success']}, 失败={batch_stats['itemcf_fail']}, "
              f"平均召回={np.mean(batch_stats['itemcf_items']) if batch_stats['itemcf_items'] else 0:.1f}个")
        print(f"    双塔: 成功={batch_stats['two_tower_success']}, 失败={batch_stats['two_tower_fail']}, "
              f"平均召回={np.mean(batch_stats['two_tower_items']) if batch_stats['two_tower_items'] else 0:.1f}个")
        
        # 定期清理
        if (batch_idx + 1) % 3 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
            print_memory_stats(f"批次{batch_idx+1}")
    
    print(f"召回完成: {len(user_recall_dict)} 用户")
    print_memory_stats("召回完成")
    
    # 4. 构建结果
    print("\n[4/5] 构建提交...")
    tst_click = pd.read_csv(str(DATA_PATH / "testA_click_log.csv"))
    tst_users = set(tst_click["user_id"].unique())
    
    user_item_score_list = []
    for user in tqdm(tst_users, desc="构建"):
        if user in user_recall_dict:
            for item, score in user_recall_dict[user]:
                user_item_score_list.append([user, item, score])
    
    recall_df = pd.DataFrame(
        user_item_score_list,
        columns=["user_id", "click_article_id", "pred_score"]
    )
    
    print(f"测试集: {len(recall_df)} 条")
    print_memory_stats("结果构建")
    
    # 5. 生成提交
    model_name = f"{MODEL_NAME}_multi_gpu"
    submit_path = submit_mod.submit(recall_df, topk=topk_submit, model_name=model_name)
    
    print(f"\n提交文件: {submit_path}")
    print_memory_stats("完成")
    print("=" * 70)
    
    return submit_path


def _load_or_build_i2i(all_click_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """加载或构建ItemCF相似度"""
    sim_path = SAVE_PATH / I2I_SIM_FILENAME
    if sim_path.exists():
        print("加载ItemCF相似度...")
        with open(sim_path, "rb") as f:
            return pickle.load(f)
    print("计算ItemCF相似度...")
    return similarity.itemcf_sim(all_click_df)

