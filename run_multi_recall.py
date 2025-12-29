#!/usr/bin/env python
"""
多路召回系统运行脚本

使用示例：
    # 使用多路召回（加权融合）
    python run_multi_recall.py --multi --fusion weighted
    
    # 使用多路召回（排名融合）
    python run_multi_recall.py --multi --fusion rank
    
    # 使用单路召回（原始 ItemCF）
    python run_multi_recall.py
    
    # 训练双塔模型
    python run_multi_recall.py --multi --train-two-tower
"""
import argparse
from src.pipeline import run


def main():
    parser = argparse.ArgumentParser(description='新闻推荐系统 - 多路召回')
    
    parser.add_argument(
        '--multi',
        action='store_true',
        help='使用多路召回（默认使用单路 ItemCF）'
    )
    
    parser.add_argument(
        '--fusion',
        type=str,
        choices=['weighted', 'rank', 'cascade'],
        default='weighted',
        help='多路召回融合模式：weighted(加权), rank(排名), cascade(级联)'
    )
    
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='提交结果的 TopK（默认 5）'
    )
    
    parser.add_argument(
        '--train-two-tower',
        action='store_true',
        help='重新训练双塔模型'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("新闻推荐系统 - 多路召回框架")
    print("=" * 70)
    print(f"配置：")
    print(f"  - 多路召回: {'是' if args.multi else '否（单路 ItemCF）'}")
    print(f"  - 融合模式: {args.fusion}")
    print(f"  - TopK: {args.topk}")
    print(f"  - 训练双塔: {'是' if args.train_two_tower else '否'}")
    print("=" * 70)
    print()
    
    # 运行
    submit_path = run(
        topk_submit=args.topk,
        use_multi_recall=args.multi,
        fusion_mode=args.fusion,
        train_two_tower=args.train_two_tower
    )
    
    print()
    print("=" * 70)
    print(f"✓ 完成！提交文件: {submit_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

