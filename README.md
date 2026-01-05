## QUICKSTART（GPU版）

### 安装

```bash
pip install -r requirements.txt
```

### 一键训练 + 召回 + 生成提交

```bash
python run_optimized.py --mode multi --train --topk 5
```

### 常用参数

- `--tt-batch`: 双塔训练 batch size（4090 建议 8192 起）
- `--tt-steps`: 每个 epoch 的训练步数（默认 2000，避免更新太少）
- `--tt-epochs`: 训练轮数
- `--no-amp`: 关闭 AMP（默认开启）


