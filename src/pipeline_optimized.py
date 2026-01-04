"""内存优化版 Pipeline - 支持分批处理和GPU加速"""
import pickle
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from time import perf_counter

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


def _log(msg: str):
    # 统一日志入口，便于后续更进一步降噪/重定向
    print(msg)


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
    # 轻量模式保留，但减少冗余输出
    t0 = perf_counter()
    
    # 1. 加载数据（只读必要字段）
    _log("加载数据...")
    all_click_df = data_mod.get_all_click_df(offline=False)
    _log(f"点击数据: {len(all_click_df)}")
    
    # 2. 基础统计
    _log("计算基础统计...")
    user_item_time_dict = data_mod.get_user_item_time(all_click_df)
    item_topk_click = data_mod.get_item_topk_click(all_click_df, k=ITEM_TOPK_K).tolist()
    
    # 获取所有用户
    all_users = all_click_df["user_id"].unique()
    _log(f"用户数: {len(all_users)}")
    
    # 3. ItemCF 相似度（如果不存在才计算）
    _log("准备ItemCF相似度...")
    i2i_sim = _load_or_build_i2i(all_click_df)
    
    # 释放不需要的数据
    del all_click_df
    gc.collect()
    # no-op
    
    # 4. 分批召回
    _log(f"分批召回: batch_size={batch_size}")
    user_recall_dict = {}
    
    n_batches = (len(all_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_users))
        batch_users = all_users[start_idx:end_idx]
        
        for user in tqdm(batch_users, desc=f"召回{batch_idx+1}/{n_batches}"):
            recs = recall.item_based_recommend(
                user_id=user,
                user_item_time_dict=user_item_time_dict,
                i2i_sim=i2i_sim,
                sim_item_topk=SIM_ITEM_TOPK,
                recall_item_num=RECALL_ITEM_NUM,
                item_topk_click=item_topk_click,
            )
            user_recall_dict[user] = recs
        
        if (batch_idx + 1) % 5 == 0:
            gc.collect()
    
    _log(f"召回完成: users={len(user_recall_dict)}")
    
    # 5. 转换为DataFrame（流式处理）
    _log("生成提交文件...")
    
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
    
    _log(f"测试集召回: {len(recall_df)}")
    
    # 生成提交
    submit_path = submit_mod.submit(recall_df, topk=topk_submit, model_name=f"{MODEL_NAME}_lite")
    
    _log(f"完成: submit={submit_path}, elapsed={perf_counter()-t0:.2f}s")
    
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
    t0 = perf_counter()
    if device != "cuda":
        device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("仅保留GPU版本：当前环境 CUDA 不可用")
    _log("启动多路召回(GPU)...")
    
    # 1. 加载数据
    _log("加载数据...")
    all_click_df = data_mod.get_all_click_df(offline=False)
    user_item_time_dict = data_mod.get_user_item_time(all_click_df)
    item_topk_click = data_mod.get_item_topk_click(all_click_df, k=ITEM_TOPK_K).tolist()
    all_users = all_click_df["user_id"].unique()
    _log(f"用户: {len(all_users)}, 点击: {len(all_click_df)}")
    
    # 2. 准备召回资源
    _log("准备召回资源...")
    
    i2i_sim = None
    if use_itemcf:
        i2i_sim = _load_or_build_i2i(all_click_df)
        _log("ItemCF 相似度已就绪")
    
    two_tower_model = None
    item_embeddings = None
    item_encoder = None
    user_features = None
    user_id_to_row = None
    item_ids_by_index = None
    
    if use_two_tower:
        if train_two_tower or not (SAVE_PATH / 'two_tower_gpu.pth').exists():
            from . import features as feat_mod
            from . import two_tower_gpu
            
            _log("训练双塔模型...")
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
            from . import two_tower_gpu
            from . import features as feat_mod
            
            two_tower_model = two_tower_gpu.load_two_tower_model_gpu(device)
            
            with open(SAVE_PATH / 'item_embeddings_gpu.pkl', 'rb') as f:
                item_embeddings = pickle.load(f)
            
            encoders = feat_mod.load_feature_encoders()
            item_encoder = encoders['item_encoder']

            # 构建用户特征矩阵（与 encoders 对齐）
            user_df = feat_mod.extract_user_features(all_click_df)
            user_feat_cols = encoders['user_feat_cols']
            user_scaler = encoders['user_scaler']
            user_features = user_scaler.transform(user_df[user_feat_cols].fillna(0)).astype(np.float32)
            user_ids = user_df['user_id'].values
            user_id_to_row = {int(uid): int(i) for i, uid in enumerate(user_ids)}

            item_ids_by_index = getattr(item_encoder, "classes_", None)
            if item_ids_by_index is None:
                item_ids_by_index = item_encoder.inverse_transform(np.arange(len(item_embeddings)))

            del user_df
            gc.collect()
            _log("双塔模型与向量已就绪")
    
    _log("资源准备完成")
    
    # 释放原始数据
    del all_click_df
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    # no-op
    
    # 3. 分批召回
    _log(f"开始召回: batch_size={batch_size}")
    user_recall_dict = {}
    
    n_batches = (len(all_users) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_users))
        batch_users = all_users[start_idx:end_idx]
        
        # 先批量做双塔召回（避免 per-user 全量 dot/sort）
        tt_batch = {}
        if use_two_tower and two_tower_model and item_embeddings is not None and user_features is not None:
            from . import two_tower_gpu
            user_hist_items_dict = {
                int(u): {item_id for item_id, _ in user_item_time_dict.get(int(u), [])}
                for u in batch_users
            }
            tt_batch = two_tower_gpu.two_tower_recall_batch(
                user_ids=[int(u) for u in batch_users],
                user_features=user_features,
                user_id_to_row=user_id_to_row,
                item_embeddings=item_embeddings,
                item_ids_by_index=item_ids_by_index,
                model=two_tower_model,
                user_hist_items_dict=user_hist_items_dict,
                recall_num=RECALL_ITEM_NUM,
                device=device,
                batch_size=max(256, min(4096, batch_size)),
                use_faiss=True,
            )

        for user in tqdm(batch_users, desc=f"召回{batch_idx+1}/{n_batches}"):
            user = int(user)
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
                except Exception as e:
                    recall_results['itemcf'] = []
                    # 静默失败：走后续兜底
            
            # 双塔召回
            if use_two_tower and two_tower_model:
                recall_results['two_tower'] = tt_batch.get(user, [])
            
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
        
        # 定期清理
        if (batch_idx + 1) % 3 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    _log(f"召回完成: users={len(user_recall_dict)}, elapsed={perf_counter()-t0:.2f}s")
    
    # 4. 构建结果
    _log("构建提交...")
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
    
    _log(f"测试集: {len(recall_df)}")
    
    # 5. 生成提交
    model_name = f"{MODEL_NAME}_multi_gpu"
    submit_path = submit_mod.submit(recall_df, topk=topk_submit, model_name=model_name)
    _log(f"完成: submit={submit_path}, total_elapsed={perf_counter()-t0:.2f}s")
    
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

