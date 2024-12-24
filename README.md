# turtle-soup-lateral-thinking-game

這是一個專為 Turtle-soup Game 設計的專案，結合多種高難度推理情境與深度學習模型，支持中文與英文兩種語言的訓練。

## 遊戲展示

[Wei-Hsu-AI/turtle-soup-lateral-thinking-game](https://huggingface.co/spaces/Wei-Hsu-AI/turtle-soup-lateral-thinking-game)

## 資料集擴增

### 資料來源

本研究使用了 [Duguce/TurtleBench1.5k](https://huggingface.co/datasets/Duguce/TurtleBench1.5k) 資料集，來自 Hugging Face，授權於 Apache License 2.0。

### 原始資料分布

資料集共包含 **1500 筆**，標籤分布如下：

-   **T（True）：46.4%** - 表示該推測為正確的答案。
-   **F（False）：42.4%** - 表示該推測為錯誤的答案。
-   **N（Irrelevant）：11.2%** - 表示該推測與問題無關。

### 資料擴增方法與比例

為了提升資料的多樣性和邏輯性，我們採用了以下四種方法進行資料擴增：

1. **Translation-based Data（翻譯生成數據）：59.0%**

    - 使用翻譯技術，例如將中文翻譯為韓文再翻譯回中文，保證語義邏輯性與內容多樣性。

2. **Manual Annotation（手動標註）：20.0%**

    - 手動生成滿足 N（Irrelevant）條件的資料，平衡標籤分布。

3. **Turtle Benchmark（原始題庫）：15.0%**

    - 專為 Turtle-soup Game 設計的題庫，涵蓋多種高難度推理情境，提供推理能力基準測試。

4. **Model Augmentation（模型生成數據）：6.0%**
    - 通過模型生成初始數據，經人工篩選增強多樣性與覆蓋範圍。

### 訓練與測試數據分布

-   **Train Dataset**：約 8000 筆
-   **Test Dataset**：約 2000 筆

測試集中的每個海龜湯故事均未出現在訓練集中。

## 訓練結果

### 模型與授權

-   使用 [ckiplab/bert-base-chinese](https://huggingface.co/ckiplab/bert-base-chinese) 作為預訓練模型（Apache License 2.0）。
-   使用 [google-bert/bert-large-uncased](https://huggingface.co/google-bert/bert-large-uncased) 作為預訓練模型（Apache License 2.0）。
-   部分實現參考 [DART](https://github.com/zjunlp/DART/tree/main) 的 GitHub 專案。

### 結果表格

| Training Method | Language | Model              | Parameters | Test Accuracy (Contrastive: Off) | Test Accuracy (Contrastive: On) |
| --------------- | -------- | ------------------ | ---------- | -------------------------------- | ------------------------------- |
| Classification  | zh       | BERT-Base-Chinese  | 102M       | 59.0%                            | 53.8%                           |
| PET             | zh       | BERT-Base-Chinese  | 102M       | 54.1%                            | 53.0%                           |
| DiffPET         | zh       | BERT-Base-Chinese  | 102M       | 54.7%                            | 50.3%                           |
| Classification  | en       | bert-large-uncased | 305M       | 65.1%                            | --                              |
| PET             | en       | bert-large-uncased | 305M       | 65.5%                            | --                              |
| DiffPET         | en       | bert-large-uncased | 305M       | 66.0%                            | --                              |

## 使用說明

專案支持多種模型訓練方法，包括 `classification`、`pet`、`diffpet` 和 `contrastive`，並支持中文和英文的配置。

### 環境

-   Python 3.11.9
-   使用 `pip install -r requirements.txt` 安裝依賴項

### 資料

-   專為 Turtle-soup Game 設計的題庫，涵蓋多種高難度推理情境。[Duguce/TurtleBench1.5k](https://huggingface.co/datasets/Duguce/TurtleBench1.5k)
-   我們的擴增後的資料集
    -   [nycu-ai113-dl-final-project/TurtleBench-extended-en](https://huggingface.co/datasets/nycu-ai113-dl-final-project/TurtleBench-extended-en)
    -   [nycu-ai113-dl-final-project/TurtleBench-extended-zh](https://huggingface.co/datasets/nycu-ai113-dl-final-project/TurtleBench-extended-zh)

### 如何使用 Hugging Face 資料集

確保已安裝 Hugging Face 的 datasets 套件：

```bash
pip install datasets
```

使用 datasets-cli 工具下載資料集到本地：

```bash
datasets-cli download nycu-ai113-dl-final-project/TurtleBench-extended-zh -d ./turtlebench_zh
```

下載完成後，資料集將儲存在指定目錄中，可以檢查下載的資料結構，例如：

```bash
ls ./turtlebench_zh
```

如需更多資訊，請參考 Hugging Face 的 CLI 工具文檔：[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)。

### 基本命令

查看所有可用參數：

```bash
python run.py -h
```

輸出範例：

```bash
usage: run.py [-h] [--config_path CONFIG_PATH] --language {zh,en} --training_function {classification,pet,diffpet,contrastive} [--save_results SAVE_RESULTS] [--save_params SAVE_PARAMS] [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to the YAML configuration file (default: config/config_zh.yaml)
  --language {zh,en}    Language to train the model (required).
  --training_function {classification,pet,diffpet,contrastive}
                        Training function to use (required).
  --save_results SAVE_RESULTS
                        Save training results to a file (default: False).
  --save_params SAVE_PARAMS
                        Save model parameters to a file (default: False).
  --device DEVICE       Device to use for training (default: "cuda" if available, else "cpu").
```

執行範例：

```bash
python run.py --language zh --training_function pet --save_results true --save_params true
```

其餘超參數：參考 `./config`
