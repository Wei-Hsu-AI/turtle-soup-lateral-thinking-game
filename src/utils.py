import matplotlib.pyplot as plt
import json
import os
from typing import Any
import matplotlib.colors as mcolors
import yaml

class AttrDict(dict):
    """支持屬性訪問的字典包裝器"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

        
def load_config(path, **kwargs):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
    if "learning_rate" in config and isinstance(config["learning_rate"], str):
        config["learning_rate"] = float(config["learning_rate"])
    if "weight_decay" in config and isinstance(config["weight_decay"], str):
        config["weight_decay"] = float(config["weight_decay"])

    config.update(kwargs)
    return AttrDict(config)

def plot_training_validation_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.title('Training and Validation Loss')
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_validation_acc(train_accuracies, val_accuracies):
    plt.figure(figsize=(8, 6))
    plt.title('Training and Validation Accuracy')
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def save_training_results(model_name: str, train_losses: list[float], train_accuracies: list[float], val_losses: list[float], val_accuracies: list[float], save_dir: str = "results"):
    """
    儲存模型的訓練結果到 JSON 檔案，包括 losses 和 accuracies。
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_results.json")
    data = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {file_path}")

def load_all_training_results(save_dir: str = "results") -> dict[str, dict[str, list[float]]]:
    """
    載入所有模型的訓練結果 JSON 檔案，返回統一的字典。
    """
    all_results = {}
    for file_name in os.listdir(save_dir):
        if file_name.endswith("_results.json"):
            model_name = file_name.replace("_results.json", "")
            file_path = os.path.join(save_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                all_results[model_name] = json.load(f)
    return all_results

def plot_losses_and_accuracies(all_results: dict[str, dict[str, list[float]]]):
    """
    繪製所有模型的 losses 和 accuracies 圖表。
    - train_losses 和 val_losses 在一張圖。
    - train_accuracies 和 val_accuracies 在另一張圖。
    """
    # 定義顏色，確保每個模型的顏色固定
    colors = list(mcolors.TABLEAU_COLORS.values())  # 預設使用 Tableau 顏色
    model_colors = {model_name: colors[i % len(colors)] for i, model_name in enumerate(all_results.keys())}
    
    # 繪製 losses 圖表
    plt.figure(figsize=(12, 6))
    for model_name, results in all_results.items():
        train_losses = results["train_losses"]
        val_losses = results["val_losses"]
        epochs = range(1, len(train_losses) + 1)
        
        train_color = model_colors[model_name]
        val_color = mcolors.to_rgba(train_color, alpha=0.6)  # 驗證使用透明的顏色
        
        plt.plot(epochs, train_losses, label=f"{model_name} - Train Loss", color=train_color)
        plt.plot(epochs, val_losses, label=f"{model_name} - Val Loss", color=val_color, linestyle="--")
    
    plt.title("Training and Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()
    
    # 繪製 accuracies 圖表
    plt.figure(figsize=(12, 6))
    for model_name, results in all_results.items():
        train_accuracies = results["train_accuracies"]
        val_accuracies = results["val_accuracies"]
        epochs = range(1, len(train_accuracies) + 1)
        
        train_color = model_colors[model_name]
        val_color = mcolors.to_rgba(train_color, alpha=0.6)  # 驗證使用透明的顏色
        
        plt.plot(epochs, train_accuracies, label=f"{model_name} - Train Accuracy", color=train_color)
        plt.plot(epochs, val_accuracies, label=f"{model_name} - Val Accuracy", color=val_color, linestyle="--")
    
    plt.title("Training and Validation Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper left', framealpha=0.8)
    plt.grid()
    plt.show()
