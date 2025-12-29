# 多路召回系统架构说明

## 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据层 (Data Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│  data.py                                                         │
│  ├─ get_all_click_df()      # 加载点击数据                       │
│  ├─ get_user_item_time()    # 用户历史序列                       │
│  └─ get_item_topk_click()   # 热门文章                           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                       特征层 (Feature Layer)                      │
├─────────────────────────────────────────────────────────────────┤
│  features.py                                                     │
│  ├─ extract_user_features()  # 用户特征：点击、活跃度            │
│  ├─ extract_item_features()  # 物品特征：热度、时间              │
│  └─ build_feature_matrix()   # 特征矩阵构建                      │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      相似度层 (Similarity Layer)                  │
├─────────────────────────────────────────────────────────────────┤
│  similarity.py          │  usercf.py                             │
│  └─ itemcf_sim()        │  └─ usercf_sim()                       │
│     物品相似度矩阵      │     用户相似度矩阵                     │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                        模型层 (Model Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│  two_tower.py                                                    │
│  ├─ TwoTowerModel              # 双塔模型类                      │
│  ├─ train_two_tower_model()    # 模型训练                        │
│  └─ two_tower_recall()         # 向量召回                        │
│                                                                   │
│  用户塔 ──→ 用户向量 ↘                                           │
│                        余弦相似度 → 匹配分数                     │
│  物品塔 ──→ 物品向量 ↗                                           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                       召回层 (Recall Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   ItemCF     │  │   UserCF     │  │   双塔模型   │          │
│  │  recall.py   │  │  usercf.py   │  │two_tower.py  │          │
│  │   权重:30%   │  │   权重:25%   │  │   权重:25%   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   热门召回   │  │  时间衰减    │  │  冷启动召回  │          │
│  │recall_strat. │  │recall_strat. │  │recall_strat. │          │
│  │   权重:10%   │  │   权重:5%    │  │   权重:5%    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                       融合层 (Fusion Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│  multi_recall.py                                                 │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  加权融合    │  │  排名融合    │  │  级联融合    │          │
│  │  weighted    │  │    RRF       │  │  cascade     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  └─→ filter_and_rerank()  # 过滤历史 + 重排序                   │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                       输出层 (Output Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│  submit.py                                                       │
│  └─ submit()          # 生成提交文件 (CSV)                       │
│     user_id | article_1 | article_2 | ... | article_K           │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    编排层 (Orchestration Layer)                   │
├─────────────────────────────────────────────────────────────────┤
│  pipeline.py                                                     │
│  ├─ run()                   # 主流程入口                         │
│  ├─ _single_recall()        # 单路召回（ItemCF）                │
│  └─ _multi_recall()         # 多路召回 + 融合                    │
│                                                                   │
│  config.py                  # 全局配置                           │
│  ├─ 路径配置                                                     │
│  ├─ 召回参数                                                     │
│  └─ 融合权重                                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心流程

### 1. 单路召回流程
```
加载数据 → 计算 ItemCF 相似度 → ItemCF 召回 → 生成提交
```

### 2. 多路召回流程
```
加载数据
  ↓
提取特征
  ↓
准备召回资源
  ├─ ItemCF 相似度矩阵
  ├─ UserCF 相似度矩阵
  └─ 双塔模型 + 物品向量库
  ↓
对每个用户执行多路召回
  ├─ ItemCF 召回
  ├─ UserCF 召回
  ├─ 双塔召回
  ├─ 热门召回
  ├─ 时间衰减召回
  └─ 冷启动召回
  ↓
融合策略（加权/排名/级联）
  ↓
过滤 + 重排序
  ↓
生成提交文件
```

---

## 数据流

```
原始数据
  ├─ train_click_log.csv    (训练集点击日志)
  └─ testA_click_log.csv    (测试集点击日志)
         ↓
  all_click_df (合并后的完整数据)
         ↓
  ┌──────────────┬──────────────┬──────────────┐
  ↓              ↓              ↓              ↓
用户历史       热门文章      用户特征      物品特征
  ↓              ↓              ↓              ↓
召回输入       热门补全      双塔训练      双塔训练
         ↓
  多路召回结果 {strategy: [(item, score), ...]}
         ↓
  融合结果 [(item, final_score), ...]
         ↓
  过滤 + TopK
         ↓
  DataFrame (user_id, click_article_id, pred_score)
         ↓
  提交文件 (user_id, article_1, article_2, ..., article_K)
```

---

## 关键接口

### pipeline.run()
```python
def run(
    topk_submit: int = 5,           # 提交 TopK
    use_multi_recall: bool = True,   # 是否多路召回
    fusion_mode: str = 'weighted',   # 融合模式
    train_two_tower: bool = False    # 是否训练双塔
) -> Path:
    """主流程入口"""
```

### 召回接口（统一格式）
```python
def xxx_recall(...) -> List[Tuple[int, float]]:
    """
    返回: [(item_id, score), ...]
    """
```

### 融合接口
```python
def hybrid_fusion(
    recall_results: Dict[str, List[Tuple[int, float]]],
    fusion_mode: str = 'weighted',
    **kwargs
) -> List[Tuple[int, float]]:
    """
    输入: {strategy_name: [(item, score), ...]}
    输出: [(item, final_score), ...]
    """
```

---

## 配置管理

### config.py 配置项

```python
# 路径配置
PROJECT_ROOT: Path          # 项目根目录
DATA_PATH: Path             # 数据目录
SAVE_PATH: Path             # 结果保存目录

# 召回参数
SIM_ITEM_TOPK: int = 10     # ItemCF 取 Top 相似物品数
RECALL_ITEM_NUM: int = 10   # 每路召回的物品数
ITEM_TOPK_K: int = 50       # 热门文章数量

# 多路召回权重
MULTI_RECALL_WEIGHTS = {
    'itemcf': 0.3,
    'usercf': 0.25,
    'two_tower': 0.25,
    'hot': 0.1,
    'time_decay': 0.05,
    'cold_start': 0.05
}

# 双塔模型配置
TWO_TOWER_EMBEDDING_DIM: int = 64
TWO_TOWER_EPOCHS: int = 10
```

---

## 扩展点

### 1. 添加新召回策略
```python
# 在 recall_strategies.py 中添加
def my_new_recall(...) -> List[Tuple[int, float]]:
    # 实现召回逻辑
    return [(item_id, score), ...]

# 在 pipeline._multi_recall() 中调用
recall_results['my_new'] = my_new_recall(...)
```

### 2. 添加新融合策略
```python
# 在 multi_recall.py 中添加
def my_fusion(recall_results, **kwargs):
    # 实现融合逻辑
    return [(item, score), ...]

# 在 hybrid_fusion() 中添加分支
if fusion_mode == 'my_fusion':
    return my_fusion(recall_results, **kwargs)
```

### 3. 添加新特征
```python
# 在 features.py 中扩展
def extract_user_features(df):
    # 添加新特征列
    user_feats.append({
        'user_id': user_id,
        ...
        'new_feature': value  # 新特征
    })
```

---

## 性能优化

### 1. 缓存机制
- ItemCF 相似度矩阵缓存：`itemcf_i2i_sim.pkl`
- UserCF 相似度矩阵缓存：`usercf_u2u_sim.pkl`
- 双塔模型缓存：`two_tower_model.pkl`
- 物品向量缓存：`item_embeddings.pkl`
- 特征编码器缓存：`feature_encoders.pkl`

### 2. 并行化机会
- 多路召回可并行执行（当前串行）
- 用户级召回可并行（多进程）
- 特征提取可向量化

### 3. 内存优化
- 使用 `reduce_mem()` 降低数据类型
- 分批处理大规模用户
- 稀疏矩阵存储相似度

---

## 依赖关系

```
pipeline.py (主编排)
  ├─ config.py
  ├─ data.py
  ├─ features.py
  │    └─ config.py
  ├─ similarity.py
  │    └─ data.py
  ├─ usercf.py
  │    └─ data.py
  ├─ two_tower.py
  │    └─ features.py
  ├─ recall.py
  ├─ recall_strategies.py
  ├─ multi_recall.py
  └─ submit.py
       └─ config.py
```

---

## 文件职责总结

| 文件 | 职责 | 核心函数 |
|------|------|----------|
| `config.py` | 全局配置 | 常量定义 |
| `data.py` | 数据加载 | `get_all_click_df()` |
| `features.py` | 特征工程 | `extract_user/item_features()` |
| `similarity.py` | ItemCF | `itemcf_sim()` |
| `usercf.py` | UserCF | `usercf_sim()`, `user_based_recommend()` |
| `two_tower.py` | 双塔模型 | `TwoTowerModel`, `two_tower_recall()` |
| `recall.py` | ItemCF召回 | `item_based_recommend()` |
| `recall_strategies.py` | 其他召回 | `hot_recall()`, `time_decay_recall()` |
| `multi_recall.py` | 融合策略 | `hybrid_fusion()` |
| `submit.py` | 提交生成 | `submit()` |
| `pipeline.py` | 流程编排 | `run()`, `_multi_recall()` |

---

## 运行时序

```
main()
  ↓
run(use_multi_recall=True)
  ↓
get_all_click_df()                    [数据加载]
  ↓
_multi_recall()
  ↓
  ├─ _load_or_build_i2i()             [ItemCF 相似度]
  ├─ _load_or_build_u2u()             [UserCF 相似度]
  └─ train_two_tower_model()          [双塔训练（可选）]
       ├─ extract_user_features()
       ├─ extract_item_features()
       └─ build_feature_matrix()
  ↓
for each user:                        [逐用户召回]
  ├─ item_based_recommend()           [ItemCF]
  ├─ user_based_recommend()           [UserCF]
  ├─ two_tower_recall()               [双塔]
  ├─ hot_recall()                     [热门]
  ├─ time_decay_recall()              [时间]
  ├─ cold_start_recall()              [冷启动]
  ↓
  hybrid_fusion()                     [融合]
  ↓
  filter_and_rerank()                 [过滤重排]
  ↓
to DataFrame                          [转换格式]
  ↓
submit()                              [生成提交]
```

---

## 总结

这是一个**高度模块化、可扩展、生产就绪**的多路召回推荐系统，具备：

- ✅ 清晰的分层架构
- ✅ 统一的接口设计
- ✅ 完善的缓存机制
- ✅ 灵活的配置管理
- ✅ 易于扩展的插件式设计

可根据业务需求快速迭代优化！

