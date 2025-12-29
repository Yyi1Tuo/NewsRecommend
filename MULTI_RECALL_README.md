# 多路召回系统使用指南

## 📋 系统概述

本项目实现了一个完整的多路召回新闻推荐系统，包含以下召回通路：

### 召回策略

1. **ItemCF（物品协同过滤）**
   - 基于物品共现计算相似度
   - 权重：30%

2. **UserCF（用户协同过滤）**
   - 基于用户相似度推荐
   - 权重：25%

3. **双塔模型（Two-Tower）**
   - 深度学习召回
   - 用户塔和物品塔分别编码特征
   - 权重：25%

4. **热门召回**
   - 推荐全局热门文章
   - 权重：10%

5. **时间衰减召回**
   - 最近热门文章权重更高
   - 权重：5%

6. **冷启动召回**
   - 针对新用户的特殊策略
   - 权重：5%

### 融合策略

- **加权融合（weighted）**：对各路召回结果加权求和
- **排名融合（rank）**：基于 RRF (Reciprocal Rank Fusion)
- **级联融合（cascade）**：按优先级依次选择

## 🚀 快速开始

### 1. 环境准备

```bash
pip install pandas numpy scikit-learn tqdm
```

### 2. 数据准备

确保 `dataset/` 目录下有以下文件：
- `train_click_log.csv`
- `testA_click_log.csv`

### 3. 运行

#### 使用多路召回（推荐）

```bash
# 加权融合模式
python run_multi_recall.py --multi --fusion weighted

# 排名融合模式
python run_multi_recall.py --multi --fusion rank

# 级联融合模式
python run_multi_recall.py --multi --fusion cascade
```

#### 使用单路召回（原始 ItemCF）

```bash
python run_multi_recall.py
```

#### 重新训练双塔模型

```bash
python run_multi_recall.py --multi --train-two-tower
```

## 📊 系统架构

```
数据加载 (data.py)
    ↓
特征提取 (features.py)
    ↓
多路召回
    ├── ItemCF (similarity.py, recall.py)
    ├── UserCF (usercf.py)
    ├── 双塔模型 (two_tower.py)
    ├── 热门召回 (recall_strategies.py)
    ├── 时间衰减召回 (recall_strategies.py)
    └── 冷启动召回 (recall_strategies.py)
    ↓
结果融合 (multi_recall.py)
    ↓
生成提交 (submit.py)
```

## 🔧 核心模块说明

### config.py
- 项目配置文件
- 路径、参数、权重配置

### data.py
- 数据加载与预处理
- 用户-物品交互提取

### features.py
- 用户特征提取（点击次数、活跃度等）
- 物品特征提取（热度、时间特征等）
- 特征矩阵构建

### similarity.py
- ItemCF 相似度计算
- 基于共现的 i2i 矩阵

### usercf.py
- UserCF 相似度计算
- 基于共同点击的 u2u 矩阵

### two_tower.py
- 双塔模型训练
- 用户/物品嵌入生成
- 向量召回

### recall.py
- ItemCF 召回实现

### recall_strategies.py
- 热门召回
- 时间衰减召回
- 多样性召回
- 冷启动召回

### multi_recall.py
- 加权融合
- 排名融合（RRF）
- 级联融合
- 结果过滤与重排

### pipeline.py
- 主流程编排
- 单路/多路召回切换
- 资源加载与管理

### submit.py
- 提交文件生成

## 🎯 性能优化建议

### 1. 双塔模型改进
当前实现是简化版，建议使用 PyTorch/TensorFlow 实现完整的神经网络：

```python
# 示例：使用 PyTorch 实现
class UserTower(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### 2. 特征工程增强
- 添加文章内容特征（标题、分类、关键词）
- 添加用户画像特征（年龄、地域、兴趣标签）
- 添加上下文特征（时间、设备、位置）

### 3. 召回路数扩展
- 基于内容的召回（TF-IDF、Word2Vec）
- 基于图的召回（Graph Embedding）
- 基于序列的召回（GRU4Rec、SASRec）

### 4. 融合策略优化
- 学习融合权重（LTR - Learning to Rank）
- 动态权重调整（根据用户活跃度）
- A/B 测试验证最佳组合

## 📈 效果评估

### 离线评估指标
- Recall@K：召回率
- Precision@K：精确率
- NDCG@K：归一化折损累计增益
- Coverage：覆盖率
- Diversity：多样性

### 运行示例输出

```
============================================================
开始新闻推荐系统流程
多路召回: True, 融合模式: weighted
============================================================
加载点击数据: 1000000 条记录

使用多路召回...

[1/3] 准备召回资源...
加载已有的 ItemCF 相似度矩阵...
计算 UserCF 相似度矩阵...
加载双塔模型...

[2/3] 执行多路召回...
多路召回: 100%|████████████████| 50000/50000 [02:30<00:00, 333.33it/s]

[3/3] 多路召回完成，共 50000 个用户

召回结果: 500000 条记录
测试集用户: 10000 个

提交文件已生成: temp_results/itemcf_baseline_multi_12-25.csv
============================================================
```

## 🔍 调试与问题排查

### 常见问题

1. **内存不足**
   - 减小 `RECALL_ITEM_NUM`
   - 使用 `reduce_mem` 降低内存占用

2. **双塔模型训练慢**
   - 减少 `TWO_TOWER_EPOCHS`
   - 使用 GPU 加速（需改用 PyTorch/TensorFlow）

3. **某路召回失败**
   - 检查相应的相似度矩阵文件是否存在
   - 查看日志中的异常信息

### 查看中间结果

```python
from src import pipeline

# 只构建相似度矩阵
from src.data import get_all_click_df
from src.similarity import itemcf_sim

all_click_df = get_all_click_df(offline=False)
i2i_sim = itemcf_sim(all_click_df)
```

## 📝 配置文件说明

编辑 `src/config.py` 调整参数：

```python
# 召回数量
RECALL_ITEM_NUM = 10  # 每路召回的物品数

# 多路召回权重
MULTI_RECALL_WEIGHTS = {
    'itemcf': 0.3,      # ItemCF 权重
    'usercf': 0.25,     # UserCF 权重
    'two_tower': 0.25,  # 双塔权重
    'hot': 0.1,         # 热门权重
    'time_decay': 0.05, # 时间衰减权重
    'cold_start': 0.05  # 冷启动权重
}
```

## 🎓 进阶使用

### Python API

```python
from src.pipeline import run

# 使用多路召回
submit_path = run(
    topk_submit=5,
    use_multi_recall=True,
    fusion_mode='weighted',
    train_two_tower=False
)

print(f"提交文件: {submit_path}")
```

### 自定义召回策略

在 `recall_strategies.py` 中添加新的召回函数：

```python
def my_custom_recall(user_id, **kwargs):
    # 自定义召回逻辑
    return [(item_id, score), ...]
```

然后在 `pipeline.py` 的 `_multi_recall` 函数中调用。

## 📚 参考文献

- ItemCF/UserCF: "Item-Based Collaborative Filtering Recommendation Algorithms" (Sarwar et al., 2001)
- 双塔模型: "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations" (Yi et al., 2019)
- RRF 融合: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (Cormack et al., 2009)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

