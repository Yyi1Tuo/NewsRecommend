from pathlib import Path


# 项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# 数据与结果目录
DATA_PATH: Path = PROJECT_ROOT / "dataset"
SAVE_PATH: Path = PROJECT_ROOT / "temp_results"

# 确保目录存在
DATA_PATH.mkdir(parents=True, exist_ok=True)
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# 召回/相似度的默认配置
SIM_ITEM_TOPK: int = 10
RECALL_ITEM_NUM: int = 10
ITEM_TOPK_K: int = 50
MODEL_NAME: str = "itemcf_baseline"

# 相似度矩阵文件名
I2I_SIM_FILENAME: str = "itemcf_i2i_sim.pkl"


