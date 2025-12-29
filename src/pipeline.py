import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
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
from . import usercf
from . import recall_strategies
from . import multi_recall
from . import submit as submit_mod
from . import features as feat_mod
from . import two_tower


def _load_or_build_i2i(all_click_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    sim_path = SAVE_PATH / I2I_SIM_FILENAME
    if sim_path.exists():
        print("加载已有的 ItemCF 相似度矩阵...")
        with open(sim_path, "rb") as f:
            return pickle.load(f)
    print("计算 ItemCF 相似度矩阵...")
    return similarity.itemcf_sim(all_click_df)


def _load_or_build_u2u(all_click_df: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """加载或构建用户相似度矩阵"""
    sim_path = SAVE_PATH / 'usercf_u2u_sim.pkl'
    if sim_path.exists():
        print("加载已有的 UserCF 相似度矩阵...")
        with open(sim_path, "rb") as f:
            return pickle.load(f)
    print("计算 UserCF 相似度矩阵...")
    return usercf.usercf_sim(all_click_df)


def _single_recall(
    all_click_df: pd.DataFrame,
    i2i_sim: Dict[int, Dict[int, float]],
    user_item_time_dict: Dict[int, List[Tuple[int, int]]],
    item_topk_click: List[int]
) -> Dict[int, List[Tuple[int, float]]]:
    """单路召回（ItemCF）"""
    user_recall_items_dict: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    
    for user in tqdm(all_click_df["user_id"].unique(), desc="ItemCF召回"):
        recs = recall.item_based_recommend(
            user_id=user,
            user_item_time_dict=user_item_time_dict,
            i2i_sim=i2i_sim,
            sim_item_topk=SIM_ITEM_TOPK,
            recall_item_num=RECALL_ITEM_NUM,
            item_topk_click=item_topk_click,
        )
        user_recall_items_dict[user] = recs
    
    return user_recall_items_dict


def _multi_recall(
    all_click_df: pd.DataFrame,
    user_item_time_dict: Dict[int, List[Tuple[int, int]]],
    item_topk_click: List[int],
    fusion_mode: str = 'weighted',
    train_two_tower: bool = False
) -> Dict[int, List[Tuple[int, float]]]:
    """多路召回并融合"""
    
    # 1. 加载/构建所有召回通路所需的资源
    print("\n[1/3] 准备召回资源...")
    
    # ItemCF
    i2i_sim = _load_or_build_i2i(all_click_df)
    
    # UserCF
    u2u_sim = _load_or_build_u2u(all_click_df)
    
    # 双塔模型
    two_tower_model = None
    item_embeddings = None
    user_features_dict = None
    item_encoder = None
    
    if train_two_tower or not (SAVE_PATH / 'two_tower_model.pkl').exists():
        print("\n准备训练双塔模型...")
        user_df = feat_mod.extract_user_features(all_click_df)
        item_df = feat_mod.extract_item_features(all_click_df)
        two_tower_model = two_tower.train_two_tower_model(all_click_df, user_df, item_df, epochs=5)
    
    # 加载双塔模型资源
    if (SAVE_PATH / 'two_tower_model.pkl').exists():
        print("加载双塔模型...")
        two_tower_model = two_tower.TwoTowerModel.load(SAVE_PATH / 'two_tower_model.pkl')
        
        with open(SAVE_PATH / 'item_embeddings.pkl', 'rb') as f:
            item_embeddings = pickle.load(f)
        
        encoders = feat_mod.load_feature_encoders()
        item_encoder = encoders['item_encoder']
        
        # 构建用户特征字典
        user_df = feat_mod.extract_user_features(all_click_df)
        user_features, _, _ = feat_mod.build_feature_matrix(all_click_df, user_df, feat_mod.extract_item_features(all_click_df))
        user_encoder = encoders['user_encoder']
        user_features_dict = {
            uid: user_features[user_encoder.transform([uid])[0]]
            for uid in all_click_df['user_id'].unique()
        }
    
    # 2. 对每个用户进行多路召回
    print("\n[2/3] 执行多路召回...")
    user_recall_items_dict: Dict[int, List[Tuple[int, float]]] = {}
    
    for user in tqdm(all_click_df["user_id"].unique(), desc="多路召回"):
        recall_results = {}
        
        # ItemCF 召回
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
        
        # UserCF 召回
        try:
            usercf_recs = usercf.user_based_recommend(
                user_id=user,
                user_item_time_dict=user_item_time_dict,
                u2u_sim=u2u_sim,
                sim_user_topk=20,
                recall_item_num=RECALL_ITEM_NUM,
                item_topk_click=item_topk_click,
            )
            recall_results['usercf'] = usercf_recs
        except Exception as e:
            recall_results['usercf'] = []
        
        # 双塔召回
        if two_tower_model and item_embeddings and user_features_dict:
            try:
                user_hist_items = {item_id for item_id, _ in user_item_time_dict.get(user, [])}
                tt_recs = two_tower.two_tower_recall(
                    user_id=user,
                    user_features_dict=user_features_dict,
                    item_embeddings=item_embeddings,
                    item_encoder=item_encoder,
                    model=two_tower_model,
                    user_hist_items=user_hist_items,
                    recall_num=RECALL_ITEM_NUM
                )
                recall_results['two_tower'] = tt_recs
            except Exception as e:
                recall_results['two_tower'] = []
        
        # 热门召回
        user_hist_items = {item_id for item_id, _ in user_item_time_dict.get(user, [])}
        hot_recs = recall_strategies.hot_recall(user_hist_items, item_topk_click, RECALL_ITEM_NUM)
        recall_results['hot'] = hot_recs
        
        # 时间衰减召回
        time_recs = recall_strategies.time_decay_recall(
            user, user_item_time_dict, all_click_df, RECALL_ITEM_NUM
        )
        recall_results['time_decay'] = time_recs
        
        # 冷启动召回
        cold_recs = recall_strategies.cold_start_recall(
            user, user_item_time_dict, item_topk_click, all_click_df, RECALL_ITEM_NUM
        )
        if cold_recs:  # 只有冷启动用户才有结果
            recall_results['cold_start'] = cold_recs
        
        # 3. 融合
        fused = multi_recall.hybrid_fusion(recall_results, fusion_mode=fusion_mode)
        
        # 过滤历史并取 TopK
        final = multi_recall.filter_and_rerank(fused, user_hist_items, topk=RECALL_ITEM_NUM * 2)
        user_recall_items_dict[user] = final
    
    print(f"\n[3/3] 多路召回完成，共 {len(user_recall_items_dict)} 个用户")
    return user_recall_items_dict


def run(
    topk_submit: int = 5,
    use_multi_recall: bool = True,
    fusion_mode: str = 'weighted',
    train_two_tower: bool = False
) -> Path:
    """
    主运行流程
    
    Args:
        topk_submit: 提交的 TopK
        use_multi_recall: 是否使用多路召回
        fusion_mode: 融合模式 ('weighted', 'rank', 'cascade')
        train_two_tower: 是否训练双塔模型
    """
    print("=" * 60)
    print("开始新闻推荐系统流程")
    print(f"多路召回: {use_multi_recall}, 融合模式: {fusion_mode}")
    print("=" * 60)
    
    # 读取全量（训练+测试）点击数据
    all_click_df = data_mod.get_all_click_df(offline=False)
    print(f"加载点击数据: {len(all_click_df)} 条记录")

    # 用户历史与热门文章
    user_item_time_dict = data_mod.get_user_item_time(all_click_df)
    item_topk_click = data_mod.get_item_topk_click(all_click_df, k=ITEM_TOPK_K).tolist()
    
    if not use_multi_recall:
        # 单路召回（原始 ItemCF）
        print("\n使用单路召回 (ItemCF)...")
        i2i_sim = _load_or_build_i2i(all_click_df)
        user_recall_items_dict = _single_recall(
            all_click_df, i2i_sim, user_item_time_dict, item_topk_click
        )
    else:
        # 多路召回
        print("\n使用多路召回...")
        user_recall_items_dict = _multi_recall(
            all_click_df,
            user_item_time_dict,
            item_topk_click,
            fusion_mode=fusion_mode,
            train_two_tower=train_two_tower
        )

    # to DataFrame
    user_item_score_list: List[Tuple[int, int, float]] = []
    for user, items in user_recall_items_dict.items():
        for item, score in items:
            user_item_score_list.append([user, item, score])
    recall_df = pd.DataFrame(user_item_score_list, columns=["user_id", "click_article_id", "pred_score"])
    print(f"\n召回结果: {len(recall_df)} 条记录")

    # 测试集过滤
    tst_click = pd.read_csv(str(DATA_PATH / "testA_click_log.csv"))
    tst_users = tst_click["user_id"].unique()
    tst_recall = recall_df[recall_df["user_id"].isin(tst_users)]
    print(f"测试集用户: {len(tst_users)} 个")

    # 生成提交
    model_name = f"{MODEL_NAME}_multi" if use_multi_recall else MODEL_NAME
    submit_path = submit_mod.submit(tst_recall, topk=topk_submit, model_name=model_name)
    print(f"\n提交文件已生成: {submit_path}")
    print("=" * 60)
    return submit_path


