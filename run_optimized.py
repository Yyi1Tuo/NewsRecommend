#!/usr/bin/env python
"""
GPU 版运行脚本（只保留 GPU 路径）

使用示例：
    # 多路召回（ItemCF + 双塔）
    python run_optimized.py --mode multi
    
    # 训练双塔模型并召回
    python run_optimized.py --mode multi --train
"""
import argparse
import sys
from pathlib import Path
import torch


def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("仅保留GPU版本：当前环境 CUDA 不可用")


def main():
    parser = argparse.ArgumentParser(description='新闻推荐系统（GPU版）')
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['multi'],
        default='multi',
        help='运行模式: multi(多路召回)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='重新训练双塔模型'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='提交结果的 TopK（默认 5）'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='批处理大小（默认 1000）'
    )

    # 双塔训练超参（in-batch listwise CE）
    parser.add_argument('--tt-epochs', type=int, default=10, help='双塔训练轮数（默认 10）')
    parser.add_argument('--tt-batch', type=int, default=8192, help='双塔训练 batch_size（默认 8192，适合 4090）')
    parser.add_argument('--tt-lr', type=float, default=1e-3, help='双塔训练学习率（默认 1e-3）')
    parser.add_argument('--tt-temp', type=float, default=0.07, help='in-batch CE temperature（默认 0.07）')
    parser.add_argument('--tt-steps', type=int, default=2000, help='每个 epoch 的训练步数（默认 2000，避免 one-epoch 现象）')
    parser.add_argument('--no-amp', action='store_true', help='关闭 AMP（默认开启，建议保留）')
    parser.add_argument('--w-itemcf', type=float, default=0.8, help='融合权重：ItemCF（默认 0.8）')
    parser.add_argument('--w-tt', type=float, default=0.2, help='融合权重：TwoTower（默认 0.2）')

    # 日志输出（后台训练建议开启）
    parser.add_argument('--log-dir', type=str, default='temp_results/logs', help='日志目录（默认 temp_results/logs）')
    parser.add_argument('--log-prefix', type=str, default='run', help='日志文件名前缀（默认 run）')
    parser.add_argument('--no-tee', action='store_true', help='不输出到终端（只写日志文件）')
    
    args = parser.parse_args()
    
    _require_cuda()

    # 设置 stdout/stderr -> 日志文件
    from src.log_utils import build_log_path, setup_std_stream_logging
    log_paths = build_log_path(
        log_dir=Path(args.log_dir),
        mode=args.mode,
        is_train=bool(args.train),
        prefix=args.log_prefix,
    )
    setup_std_stream_logging(log_paths.log_file, tee_to_console=(not args.no_tee))
    print(f"[log] {log_paths.log_file}")
    
    # 运行
    from src.pipeline_optimized import run_multi_gpu
    submit_path = run_multi_gpu(
        topk_submit=args.topk,
        use_itemcf=True,
        use_two_tower=True,
        train_two_tower=args.train,
        batch_size=args.batch_size,
        device='cuda',
        tt_epochs=args.tt_epochs,
        tt_batch_size=args.tt_batch,
        tt_lr=args.tt_lr,
        tt_temperature=args.tt_temp,
        tt_steps_per_epoch=args.tt_steps,
        tt_use_amp=(not args.no_amp),
        weight_itemcf=args.w_itemcf,
        weight_two_tower=args.w_tt,
    )
    print(f"完成: {submit_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

