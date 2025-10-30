from src.pipeline import run


if __name__ == "__main__":
    submit_path = run(topk_submit=5)
    print(f"提交文件已生成: {submit_path}")