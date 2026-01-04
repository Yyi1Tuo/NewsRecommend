"""
保留兼容入口：转发到 GPU 优化版 pipeline。
建议直接使用 `run_optimized.py`。
"""

from src.pipeline_optimized import run_multi_gpu


if __name__ == "__main__":
    submit_path = run_multi_gpu(topk_submit=5, train_two_tower=False, device="cuda")
    print(f"提交文件已生成: {submit_path}")