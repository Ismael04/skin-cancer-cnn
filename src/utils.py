# src/utils.py
import random, numpy as np, torch

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ces deux lignes rendent l'exécution un peu plus lente mais plus stable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
