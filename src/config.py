from pathlib import Path
import torch

# chemins
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GAMES_DIR = DATA_DIR / "game_moves"
MODEL_DIR = DATA_DIR / "model"
MODEL_DIR_9X9 = MODEL_DIR / "model_9x9"
# LOG_DIR = DATA_DIR / "logs"

MODEL_DIR_9X9.mkdir(parents=True, exist_ok=True)
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)