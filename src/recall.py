from typing import Dict, List, Tuple


def item_based_recommend(
    user_id: int,
    user_item_time_dict: Dict[int, List[Tuple[int, int]]],
    i2i_sim: Dict[int, Dict[int, float]],
    sim_item_topk: int,
    recall_item_num: int,
    item_topk_click: List[int],
):
    """
    基于文章协同过滤的召回，返回 [(item, score), ...]
    """
    # 取出用户历史点击记录，防止后面重复推荐
    user_hist_items = user_item_time_dict[user_id]
    user_hist_item_ids = {item_id for item_id, _ in user_hist_items}

    item_rank: Dict[int, float] = {}
    for _, (i, _) in enumerate(user_hist_items):
        if i not in i2i_sim:#如果文章不在相似度矩阵中则跳过（例如新物品）
            continue
        ##取最相似的 TopK 物品
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_item_ids:
                continue
            item_rank.setdefault(j, 0.0)
            item_rank[j] += wij

    # 不足 recall_item_num，用热门补全
    if len(item_rank) < recall_item_num:
        for idx, item in enumerate(item_topk_click):
            if item in item_rank:  # 修复：判断是否在 key 中
                continue
            item_rank[item] = -idx - 100.0
            if len(item_rank) == recall_item_num:
                break

    item_rank_sorted = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
    return item_rank_sorted


