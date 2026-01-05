"""双塔模型 GPU 优化版本 - 使用 PyTorch"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Sequence
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .config import SAVE_PATH


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


def _maybe_to_faiss_gpu_index(index, device: int = 0):
    """
    如果 faiss-gpu 可用，则把 CPU index 转成 GPU index。
    faiss-cpu 环境下会直接返回原 index。
    """
    faiss = _try_import_faiss()
    if faiss is None:
        return index
    if not hasattr(faiss, "StandardGpuResources"):
        return index
    try:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, device, index)
    except Exception:
        return index


def build_or_load_faiss_index(
    item_embeddings: np.ndarray,
    index_path: Path,
    use_gpu: bool = True,
    gpu_device: int = 0
):
    """
    基于 item_embeddings 构建/加载 Faiss Index（IP，相似度=点积；向量已做 L2 normalize 时等价余弦）。
    - 若 faiss 不可用：返回 None（上层走 torch.topk 兜底）
    - 自动用文件 mtime 判断 index 是否过期
    """
    faiss = _try_import_faiss()
    if faiss is None:
        return None

    emb = np.asarray(item_embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"item_embeddings must be 2D, got {emb.shape}")

    # 简单过期判断：index 文件不存在或比 embeddings pkl 更老时重建
    need_rebuild = (not index_path.exists())
    if not need_rebuild:
        try:
            need_rebuild = index_path.stat().st_mtime < (SAVE_PATH / 'item_embeddings_gpu.pkl').stat().st_mtime
        except Exception:
            need_rebuild = False

    if not need_rebuild:
        try:
            index = faiss.read_index(str(index_path))
            return _maybe_to_faiss_gpu_index(index, device=gpu_device) if use_gpu else index
        except Exception:
            need_rebuild = True

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    try:
        faiss.write_index(index, str(index_path))
    except Exception:
        # 写失败不影响运行
        pass

    return _maybe_to_faiss_gpu_index(index, device=gpu_device) if use_gpu else index


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


class PosPairDataset(Dataset):
    """正样本对数据集：仅 (user_feat, item_feat)，用于 in-batch negatives 的 listwise 训练"""
    def __init__(self, pos_pairs: np.ndarray, user_features: np.ndarray, item_features: np.ndarray):
        # pos_pairs: shape (N,2) -> (user_idx, item_idx)
        self.pos_pairs = pos_pairs.astype(np.int64, copy=False)
        self.user_features = user_features.astype(np.float32, copy=False)
        self.item_features = item_features.astype(np.float32, copy=False)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        u, i = self.pos_pairs[idx]
        return (
            torch.from_numpy(self.user_features[int(u)]),
            torch.from_numpy(self.item_features[int(i)]),
        )


class MLPTower(nn.Module):
    """通用 MLP Tower（更深、更稳定：LayerNorm + Dropout）"""
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers: List[nn.Module] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)


class UserTower(MLPTower):
    """用户塔"""
    pass


class ItemTower(MLPTower):
    """物品塔"""
    pass


class TwoTowerModelGPU(nn.Module):
    """双塔模型 GPU版本"""
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embedding_dim: int = 64,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
        num_users: int = 0,
        num_items: int = 0,
        user_id_emb_dim: int = 64,
        item_id_emb_dim: int = 250,
        item_id_init: Optional[np.ndarray] = None,
    ):
        super().__init__()
        # ID Embeddings（真正的双塔核心信号）
        if num_users <= 0 or num_items <= 0:
            raise ValueError("num_users/num_items must be provided for ID-embedding two-tower")
        self.user_id_emb = nn.Embedding(num_users, user_id_emb_dim)
        self.item_id_emb = nn.Embedding(num_items, item_id_emb_dim)
        if item_id_init is not None:
            init = np.asarray(item_id_init, dtype=np.float32)
            if init.shape != (num_items, item_id_emb_dim):
                raise ValueError(f"item_id_init shape {init.shape} != ({num_items},{item_id_emb_dim})")
            with torch.no_grad():
                self.item_id_emb.weight.copy_(torch.from_numpy(init))

        # 数值特征分支（小型 MLP 到中间维度）
        self.user_num_proj = nn.Sequential(
            nn.Linear(user_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.item_num_proj = nn.Sequential(
            nn.Linear(item_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ID embedding 投影（将 250-d articles_emb 压到更合适的融合维度）
        self.user_id_proj = nn.Sequential(
            nn.Linear(user_id_emb_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        self.item_id_proj = nn.Sequential(
            nn.Linear(item_id_emb_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        # 融合后的 tower（concat 后再 MLP -> embedding_dim）
        self.user_tower = UserTower(64 + 64, embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.item_tower = ItemTower(128 + 64, embedding_dim=embedding_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.embedding_dim = embedding_dim
    
    def encode_user(self, user_idx: torch.Tensor, user_num_feat: torch.Tensor) -> torch.Tensor:
        uid = self.user_id_proj(self.user_id_emb(user_idx))
        unum = self.user_num_proj(user_num_feat)
        return self.user_tower(torch.cat([uid, unum], dim=1))

    def encode_item(self, item_idx: torch.Tensor, item_num_feat: torch.Tensor) -> torch.Tensor:
        iid = self.item_id_proj(self.item_id_emb(item_idx))
        inum = self.item_num_proj(item_num_feat)
        return self.item_tower(torch.cat([iid, inum], dim=1))

    def forward(self, user_idx, user_num_feat, item_idx, item_num_feat):
        user_emb = self.encode_user(user_idx, user_num_feat)
        item_emb = self.encode_item(item_idx, item_num_feat)
        # 余弦相似度（已归一化，直接点积）
        score = torch.sum(user_emb * item_emb, dim=1)
        return score
    
    def get_user_embedding(self, user_idx: torch.Tensor, user_num_feat: torch.Tensor):
        """获取用户向量（推理时建议外层包 torch.no_grad()）"""
        return self.encode_user(user_idx, user_num_feat)
    
    def get_item_embedding(self, item_idx: torch.Tensor, item_num_feat: torch.Tensor):
        """获取物品向量（推理时建议外层包 torch.no_grad()）"""
        return self.encode_item(item_idx, item_num_feat)


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
    # 减少冗余输出：只保留关键日志
    
    # 构建训练数据（正样本）
    click_data = encoders['click_with_features']
    pos_interactions = click_data[['user_idx', 'item_idx']].values
    pos_labels = np.ones(len(pos_interactions))
    
    # print(f"正样本数量: {len(pos_interactions)}")
    
    # 负采样（内存优化：采样等量负样本，避免真实正样本）
    n_users = len(user_features)
    n_items = len(item_features)
    n_neg = min(len(pos_interactions), 100000)  # 增加负样本数量以提升训练效果
    
    # 构建正样本集合用于过滤
    pos_set = set(map(tuple, pos_interactions))
    # print(f"构建正样本哈希表: {len(pos_set)} 个正样本")
    
    # 负采样（过滤掉真实正样本）
    neg_samples = []
    attempts = 0
    max_attempts = n_neg * 10
    
    # print("开始负采样...")
    while len(neg_samples) < n_neg and attempts < max_attempts:
        batch_size_neg = min(10000, n_neg - len(neg_samples))
        neg_users = np.random.randint(0, n_users, batch_size_neg)
        neg_items = np.random.randint(0, n_items, batch_size_neg)
        
        for u, i in zip(neg_users, neg_items):
            if (u, i) not in pos_set:
                neg_samples.append([u, i])
                if len(neg_samples) >= n_neg:
                    break
        
        attempts += batch_size_neg
        # 采样过程不刷屏
    
    neg_interactions = np.array(neg_samples)
    neg_labels = np.zeros(len(neg_interactions))
    
    # print(f"负采样完成: {len(neg_interactions)} 个负样本")
    
    # 合并正负样本
    all_interactions = np.vstack([
        np.column_stack([pos_interactions, pos_labels]),
        np.column_stack([neg_interactions, neg_labels])
    ])
    
    # 打乱
    np.random.shuffle(all_interactions)
    
    # 验证标签分布
    labels_check = all_interactions[:, 2]
    # print(f"标签分布: 正样本={np.sum(labels_check == 1)}, 负样本={np.sum(labels_check == 0)}")
    
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
    print(f"开始训练双塔模型: epochs={epochs}, batch_size={batch_size}, device={device}")
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        pos_scores = []
        neg_scores = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (user_feat, item_feat, labels) in enumerate(pbar):
            user_feat = user_feat.to(device)
            item_feat = item_feat.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            scores = model(user_feat, item_feat)
            loss = criterion(scores, labels)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 记录正负样本分数
            with torch.no_grad():
                pos_mask = labels > 0.5
                neg_mask = labels < 0.5
                if pos_mask.sum() > 0:
                    pos_scores.extend(scores[pos_mask].cpu().numpy().tolist())
                if neg_mask.sum() > 0:
                    neg_scores.extend(scores[neg_mask].cpu().numpy().tolist())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'pos_score': f'{scores[pos_mask].mean().item():.4f}' if pos_mask.sum() > 0 else 'N/A',
                'neg_score': f'{scores[neg_mask].mean().item():.4f}' if neg_mask.sum() > 0 else 'N/A'
            })
            
            # 不打印 batch 级别明细
        
        avg_loss = total_loss / batch_count
        avg_pos_score = np.mean(pos_scores) if pos_scores else 0
        avg_neg_score = np.mean(neg_scores) if neg_scores else 0
        
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, pos={avg_pos_score:.4f}, neg={avg_neg_score:.4f}")
    
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
    
    print(f"物品向量已保存: shape={item_embeddings.shape}")
    
    return model


def train_two_tower_inbatch_gpu(
    all_click_df: pd.DataFrame,
    user_features: np.ndarray,
    item_features: np.ndarray,
    encoders: Dict,
    epochs: int = 5,
    batch_size: int = 2048,
    device: str = "cuda",
    lr: float = 1e-3,
    temperature: float = 0.07,
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.2,
    symmetric_loss: bool = False,
    steps_per_epoch: Optional[int] = None,
    use_amp: bool = True,
) -> TwoTowerModelGPU:
    """
    In-batch negatives 的 listwise 训练（交叉熵）：
    logits = (u_emb @ v_emb^T) / temperature, target=diag
    """
    # 关键：listwise in-batch CE 要求 batch 内的对角配对是“唯一正样本”。
    # 但真实数据里同一 user 会出现多次；若同一 user 在同一 batch 出现多个不同 item，
    # 会把另外一个“真正的正样本”当作负样本，造成强噪声，导致完全不收敛。
    # 解决：每个 batch 采样唯一 user，每个 user 随机采样一个正样本 item。
    click_data = encoders["click_with_features"][["user_idx", "item_idx"]].drop_duplicates()
    user_to_items: Dict[int, np.ndarray] = (
        click_data.groupby("user_idx")["item_idx"].apply(lambda x: x.values).to_dict()
    )
    unique_users = np.fromiter(user_to_items.keys(), dtype=np.int64)
    if len(unique_users) < batch_size:
        raise ValueError(f"unique users ({len(unique_users)}) < batch_size ({batch_size})")

    # item embedding 用 articles_emb 初始化（250维）
    try:
        from .features import load_or_build_articles_emb_matrix
        item_ids_by_index = np.asarray(encoders["item_encoder"].classes_, dtype=np.int64)
        item_id_init = load_or_build_articles_emb_matrix(item_ids_by_index=item_ids_by_index)
    except Exception:
        item_id_init = None

    model = TwoTowerModelGPU(
        user_dim=user_features.shape[1],
        item_dim=item_features.shape[1],
        embedding_dim=64,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_users=len(encoders["user_encoder"].classes_),
        num_items=len(encoders["item_encoder"].classes_),
        user_id_emb_dim=64,
        item_id_emb_dim=250,
        item_id_init=item_id_init,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    if steps_per_epoch is None:
        # 不要让 epoch 变成“一个 epoch 只有很少 step”的形式；默认给足更新次数
        steps_per_epoch = max(500, int(len(click_data) / batch_size))
    print(
        f"开始训练双塔(in-batch CE): epochs={epochs}, batch_size={batch_size}, "
        f"steps/epoch={steps_per_epoch}, temp={temperature}, lr={lr}, device={device}"
    )
    # torch.cuda.amp.* 已逐步弃用，使用 torch.amp.*（兼容 CUDA AMP）
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        diag_mean = 0.0
        off_max_mean = 0.0
        acc1 = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for _ in pbar:
            # 采样 batch_size 个唯一 user
            batch_users = np.random.choice(unique_users, size=batch_size, replace=False)
            # 每个 user 随机采样一个正样本 item
            batch_items = np.empty(batch_size, dtype=np.int64)
            for k, u in enumerate(batch_users):
                items = user_to_items[int(u)]
                batch_items[k] = int(items[np.random.randint(0, len(items))])

            user_idx = torch.from_numpy(batch_users).to(device=device, dtype=torch.long)
            item_idx = torch.from_numpy(batch_items).to(device=device, dtype=torch.long)
            user_num = torch.from_numpy(user_features[batch_users].astype(np.float32, copy=False)).to(device)
            item_num = torch.from_numpy(item_features[batch_items].astype(np.float32, copy=False)).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp and device == "cuda"):
                u = model.get_user_embedding(user_idx, user_num)   # (B,D)
                v = model.get_item_embedding(item_idx, item_num)   # (B,D)

                logits = torch.matmul(u, v.t()) / temperature
                target = torch.arange(logits.size(0), device=device)

                loss = ce(logits, target)
                if symmetric_loss:
                    loss = 0.5 * (loss + ce(logits.t(), target))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                steps += 1
                total_loss += loss.item()
                diag = torch.diagonal(logits)
                diag_mean += diag.mean().item()
                # 每行最大负样本（排除对角）
                neg = logits.clone()
                # AMP 下 logits 可能是 fp16/bf16，不能用 -1e9（会溢出），用该 dtype 可表示的最小值
                neg.fill_diagonal_(torch.finfo(neg.dtype).min)
                off_max = neg.max(dim=1).values
                off_max_mean += off_max.mean().item()
                pred = logits.argmax(dim=1)
                acc1 += (pred == target).float().mean().item()

                pbar.set_postfix({
                    "loss": f"{(total_loss/steps):.4f}",
                    "pos": f"{(diag_mean/steps):.3f}",
                    "neg_max": f"{(off_max_mean/steps):.3f}",
                    "acc@1": f"{(acc1/steps):.3f}",
                })

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"loss={total_loss/max(steps,1):.4f}, "
            f"pos={diag_mean/max(steps,1):.3f}, "
            f"neg_max={off_max_mean/max(steps,1):.3f}, "
            f"acc@1={acc1/max(steps,1):.3f}"
        )

    print("双塔模型训练完成(in-batch)")
    torch.save(model.state_dict(), SAVE_PATH / "two_tower_gpu.pth")

    # 预计算物品向量
    model.eval()
    item_embeddings = []
    item_idx_all = torch.arange(len(item_features), device=device, dtype=torch.long)
    item_num_all = torch.FloatTensor(item_features).to(device)
    with torch.no_grad():
        for i in range(0, len(item_num_all), batch_size):
            idx_b = item_idx_all[i:i + batch_size]
            num_b = item_num_all[i:i + batch_size]
            emb = model.get_item_embedding(idx_b, num_b).cpu().numpy()
            item_embeddings.append(emb)
    item_embeddings = np.vstack(item_embeddings)
    with open(SAVE_PATH / "item_embeddings_gpu.pkl", "wb") as f:
        pickle.dump(item_embeddings, f)
    print(f"物品向量已保存: shape={item_embeddings.shape}")
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
        # 兼容旧接口：dict 里存的是 (user_idx, user_num_feat)
        user_idx, user_num = user_features_dict[user_id]
        user_idx_t = torch.tensor([int(user_idx)], device=device, dtype=torch.long)
        user_num_t = torch.as_tensor(user_num, device=device, dtype=torch.float32).unsqueeze(0)
        user_emb = model.get_user_embedding(user_idx_t, user_num_t).cpu().numpy()[0]
        
        # 与所有物品计算相似度（原始实现：每用户全量 dot + 全排序，较慢）
        # 仍保留以兼容旧代码；推荐改用 two_tower_recall_batch_* 接口。
        scores = np.dot(item_embeddings, user_emb)
        top_indices = np.argsort(scores)[::-1]
        classes = getattr(item_encoder, "classes_", None)
        results = []
        for idx in top_indices:
            item_id = int(classes[idx]) if classes is not None else item_encoder.inverse_transform([idx])[0]
            if item_id not in user_hist_items:
                results.append((item_id, float(scores[idx])))
                if len(results) >= recall_num:
                    break
        
        return results


def two_tower_recall_batch(
    user_ids: Iterable[int],
    user_features: np.ndarray,
    user_id_to_row: Dict[int, int],
    item_embeddings: np.ndarray,
    item_ids_by_index: np.ndarray,
    model: TwoTowerModelGPU,
    user_hist_items_dict: Dict[int, set],
    recall_num: int = 10,
    device: str = "cuda",
    batch_size: int = 4096,
    use_faiss: bool = True,
    faiss_index_path: Optional[Path] = None,
    faiss_gpu_device: int = 0,
    extra_candidates: int = 50
) -> Dict[int, List[Tuple[int, float]]]:
    """
    批量双塔召回：
    1) 批量算用户 embedding（GPU）
    2) Faiss (优先) 或 torch.topk（兜底）做 topK 检索
    """
    uids = list(user_ids)
    if not uids:
        return {}

    model.eval()
    item_emb = np.asarray(item_embeddings, dtype=np.float32)
    item_ids_by_index = np.asarray(item_ids_by_index)
    k_search = int(recall_num + max(0, extra_candidates))

    # 准备检索器
    index = None
    if use_faiss:
        idx_path = faiss_index_path or (SAVE_PATH / "faiss_item.index")
        index = build_or_load_faiss_index(
            item_embeddings=item_emb,
            index_path=idx_path,
            use_gpu=(device == "cuda"),
            gpu_device=faiss_gpu_device
        )

    results: Dict[int, List[Tuple[int, float]]] = {}

    # torch.topk 兜底：把 item embeddings 放到 GPU（一次）以便批量检索
    item_emb_t = None
    if index is None:
        item_emb_t = torch.from_numpy(item_emb).to(device)

    with torch.no_grad():
        for i in range(0, len(uids), batch_size):
            batch_uids = uids[i:i + batch_size]
            rows = [user_id_to_row.get(uid, -1) for uid in batch_uids]
            # 过滤没有特征的用户
            valid = [(uid, r) for uid, r in zip(batch_uids, rows) if r >= 0]
            if not valid:
                continue

            v_uids, v_rows = zip(*valid)
            # 兼容两种模式：
            # - 旧：row 是 user_features 的行索引
            # - 新：row 既是数值特征行索引，也是 user_id embedding 的 index（通过严格排序对齐实现）
            u_idx = torch.from_numpy(np.array(v_rows, dtype=np.int64)).to(device=device, dtype=torch.long)
            u_num = torch.from_numpy(user_features[np.array(v_rows, dtype=np.int64)].astype(np.float32, copy=False)).to(device)
            u_emb = model.get_user_embedding(u_idx, u_num)  # (B, D) 已normalize
            u_emb_np = u_emb.detach().cpu().numpy().astype(np.float32)

            if index is not None:
                scores, idxs = index.search(u_emb_np, k_search)  # (B, k)
            else:
                # (B, D) x (D, N) -> (B, N)
                sim = torch.matmul(u_emb, item_emb_t.t())
                top_scores, top_idx = torch.topk(sim, k=k_search, dim=1, largest=True, sorted=True)
                scores = top_scores.detach().cpu().numpy()
                idxs = top_idx.detach().cpu().numpy()

            for b, uid in enumerate(v_uids):
                hist = user_hist_items_dict.get(uid, set())
                recs: List[Tuple[int, float]] = []
                for j, s in zip(idxs[b].tolist(), scores[b].tolist()):
                    item_id = int(item_ids_by_index[j])
                    if item_id in hist:
                        continue
                    recs.append((item_id, float(s)))
                    if len(recs) >= recall_num:
                        break
                results[int(uid)] = recs

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
        
        model = TwoTowerModelGPU(
            user_dim=user_dim,
            item_dim=item_dim,
            embedding_dim=64,
            hidden_dims=(256, 128),
            dropout=0.2,
            num_users=len(encoders["user_encoder"].classes_),
            num_items=len(encoders["item_encoder"].classes_),
            user_id_emb_dim=64,
            item_id_emb_dim=250,
            item_id_init=None,  # 权重会被 checkpoint 覆盖；无需在加载时重复读 1GB CSV
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        print(f"加载双塔模型失败（可能需要重训）: {e}")
        return None

