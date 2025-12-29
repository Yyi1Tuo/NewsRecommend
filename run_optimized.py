#!/usr/bin/env python
"""
内存优化版运行脚本 - 支持GPU加速

使用示例：
    # 轻量级模式（仅ItemCF，最省内存）
    python run_optimized.py --mode lite
    
    # 多路召回（ItemCF + 双塔，GPU加速）
    python run_optimized.py --mode multi --gpu
    
    # 训练双塔模型
    python run_optimized.py --mode multi --gpu --train
"""
import argparse
import sys
import torch


def check_environment():
    """检查环境配置"""
    print("=" * 70)
    print("环境检查")
    print("=" * 70)
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
    
    # 检查内存
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"系统内存: {mem.total / 1024**3:.1f} GB")
        print(f"可用内存: {mem.available / 1024**3:.1f} GB")
    except ImportError:
        print("未安装 psutil，无法检查系统内存")
    
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description='新闻推荐系统 - 内存优化版')
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['lite', 'multi'],
        default='lite',
        help='运行模式: lite(轻量级ItemCF) 或 multi(多路召回)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='使用GPU加速（如果可用）'
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
    
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='仅检查环境，不运行'
    )
    
    args = parser.parse_args()
    
    # 检查环境
    check_environment()
    
    if args.check_env:
        return
    
    # 配置
    print("运行配置：")
    print(f"  模式: {args.mode}")
    print(f"  GPU: {'是' if args.gpu else '否'}")
    print(f"  TopK: {args.topk}")
    print(f"  批大小: {args.batch_size}")
    if args.mode == 'multi':
        print(f"  训练双塔: {'是' if args.train else '否'}")
    print()
    
    # 检查GPU
    if args.gpu and not torch.cuda.is_available():
        print("警告: 指定使用GPU但CUDA不可用，将使用CPU")
        args.gpu = False
    
    # 运行
    if args.mode == 'lite':
        print("启动轻量级模式（仅ItemCF）...")
        from src.pipeline_optimized import run_lightweight
        
        submit_path = run_lightweight(
            topk_submit=args.topk,
            use_gpu=args.gpu,
            batch_size=args.batch_size
        )
    
    elif args.mode == 'multi':
        print("启动多路召回模式（ItemCF + 双塔）...")
        from src.pipeline_optimized import run_multi_gpu
        
        device = 'cuda' if args.gpu else 'cpu'
        submit_path = run_multi_gpu(
            topk_submit=args.topk,
            use_itemcf=True,
            use_two_tower=True,
            train_two_tower=args.train,
            batch_size=args.batch_size,
            device=device
        )
    
    print()
    print("=" * 70)
    print(f"✓ 完成！提交文件: {submit_path}")
    print("=" * 70)


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

