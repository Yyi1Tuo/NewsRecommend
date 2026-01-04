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
    
    args = parser.parse_args()
    
    _require_cuda()
    
    # 运行
    from src.pipeline_optimized import run_multi_gpu
    submit_path = run_multi_gpu(
        topk_submit=args.topk,
        use_itemcf=True,
        use_two_tower=True,
        train_two_tower=args.train,
        batch_size=args.batch_size,
        device='cuda'
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

