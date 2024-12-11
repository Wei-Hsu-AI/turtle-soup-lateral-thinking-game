import json
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
from pathlib import Path
import torch

class TurtleBenchDataset(Dataset):
    def __init__(self, data_path, prompt_path, max_length=512):
        """
        初始化 Dataset
        :param data_path: JSON 檔案路徑
        :param prompt_path: Prompt 檔案路徑
        :param max_length: 最大序列長度
        """
        self.data_path = Path(data_path)
        self.prompt_path = Path(prompt_path)
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.max_length = max_length

        # 載入資料
        self.data = self._load_data()

    def _load_data(self):
        """
        載入 JSON 檔案與 Prompt 檔案，根據輸入數據的長度動態選擇適合的 Prompt，
        並生成包含填充後 Prompt 和模板的處理後數據。

        Returns:
            List[dict]: 處理後的數據列表，每筆資料包含 `input_text` 和 `label`。
        """

        # 載入數據檔案 (包含 surface, bottom, user_guess, label)
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 載入 Prompt 檔案，包含 short, medium, long Prompt 的結構
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        short_prompt = prompts["short"]
        medium_prompt = prompts["medium"]
        long_prompt = prompts["long"]

        processed_data = []
        for entry in data:
            # 提取數據字段
            surface = entry["surface"]  # 湯面部分
            bottom = entry["bottom"]    # 湯底部分
            user_guess = entry["user_guess"]  # 玩家猜測

            #  將 lable 都填充為一樣長度，這裡不選擇 [PAD] 是因為需要計算 loss
            label_map = {
                "T": "正確[SEP]",
                "F": "錯誤[SEP]",
                "N": "不知道"
            }
            label = label_map[entry["label"]]  # 將 label 映射為中文

            template = f"根據判定規則，此玩家的猜測為[MASK][MASK][MASK]"

            data_length = len(surface) + len(bottom) + len(user_guess)
            prompt_filled = ""
        
            # 判斷能否使用 long_prompt
            if data_length + long_prompt["length"] + len(template) <= self.max_length:
                prompt_filled = long_prompt["prompt"].format(surface=surface, bottom=bottom)

            # 判斷能否使用 medium_prompt
            elif data_length + medium_prompt["length"] + len(template) <= self.max_length:
                prompt_filled = medium_prompt["prompt"].format(surface=surface, bottom=bottom)

            # 判斷能否使用 short_prompt
            elif data_length + short_prompt["length"] + len(template) <= self.max_length:
                prompt_filled = short_prompt["prompt"].format(surface=surface, bottom=bottom)

            else:
                # 如果所有 Prompt 加數據的總長度都超過 max_length，跳過該數據
                continue

            # 合成最終的輸入文本，包含填充後 Prompt、玩家猜測和模板
            input_text = prompt_filled + user_guess + '[SEP]' + template
            processed_data.append({"input_text": input_text, "label": label})

        return processed_data

    def __len__(self):
        """返回資料長度"""
        return len(self.data)

    def __getitem__(self, idx):
        """處理每筆資料"""
        item = self.data[idx]
        input_text = item["input_text"]
        label = item["label"]

        # Tokenize 輸入和標籤
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 將 label 轉換為詞彙 ID
        label_tokens = self.tokenizer(
            label,
            padding="max_length",
            truncation=True,
            max_length=3,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].squeeze(0)  # 取出 ID 並壓縮維度

        labels = torch.full(inputs["input_ids"].shape, -100)

        # 獲取 [MASK] 的位置
        mask_indices = (inputs["input_ids"] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # 填充 [MASK] 的位置
        for i, mask_idx in enumerate(mask_indices):
            labels[0, mask_idx] = label_tokens[i]

        # 輸出包含 input_ids, attention_mask 和 label_id
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }

