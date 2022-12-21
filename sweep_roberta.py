import wandb
from numpy import random
import os

def train():
    #Add wandb project
    command = f"PYTHONPATH=src python3 src/scripts/train_predict_model.py --is_sweeping"
    os.system(command)

train()
