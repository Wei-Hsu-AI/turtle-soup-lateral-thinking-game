import json
from torch.utils.data import Dataset
from pathlib import Path
import torch

class TurtleSoupDataset(Dataset):
    def __init__(self, data_path, prompt_path, tokenizer, template, label_map, max_length=512):
        """
        初始化 Dataset
        :param data_path: JSON 檔案路徑
        :param prompt_path: Prompt 檔案路徑
        :param template: 模板字符串
        :param label_map: 標籤映射字典
        :param max_length: 最大序列長度
        """
        self.data_path = Path(data_path)
        self.prompt_path = Path(prompt_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.template = template
        self.label_map = label_map

        self.prompts = self._load_prompts()
        self.data = self._load_data()


    def _load_prompts(self):
        """
        載入 Prompt 檔案並 Tokenize 計算長度
        :return: 包含 Tokenized 長度的 Prompt 字典
        """
        # 載入 Prompt 檔案，包含 short, medium, long Prompt 的結構
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        tokenized_prompts = {}
        for key, value in prompts.items():
            prompt_text = value["prompt"].format(surface="", bottom="")
            tokenized_length = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

            tokenized_prompts[key] = {
                "prompt": value["prompt"],
                "length": tokenized_length
            }

        return tokenized_prompts

    def _load_data(self):
        """
        載入 JSON 檔案，根據輸入數據的長度動態選擇適合的 Prompt，
        並生成包含填充後 Prompt 和模板的處理後數據。

        Returns:
            List[dict]: 處理後的數據列表，每筆資料包含 `input_text` 和 `label`。
        """

        # 載入數據檔案 (包含 surface, bottom, user_guess, label)
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        template_length = len(self.tokenizer.encode(self.template, add_special_tokens=False))

        processed_data = []
        for entry in data:
            # 提取數據字段
            surface = entry["surface"]  # 湯面部分
            bottom = entry["bottom"]    # 湯底部分
            user_guess = entry["user_guess"]  # 玩家猜測

            label = self._map_label(entry["label"])
            prompt_filled = self._select_prompt(surface, bottom, user_guess, template_length)

            # 如果所有 Prompt 加數據的總長度都超過 max_length，跳過該數據
            if not prompt_filled:
                continue

            # 合成最終的輸入文本，包含填充後 Prompt、玩家猜測和模板
            input_text = prompt_filled + self.template

            # 將 label 轉換為詞彙 ID
            label_id = self.tokenizer.encode(
                label,
                add_special_tokens=False
            )[0]

            processed_data.append({"input_text": input_text, "label_id": label_id})

        return processed_data
    
    def _map_label(self, label):
        """映射標籤為對應的中文描述"""
        return self.label_map[label]
    
    def _select_prompt(self, surface, bottom, user_guess, template_length):
        """
        根據數據長度選擇適合的 Prompt
        :param surface: 湯面部分文本
        :param bottom: 湯底部分文本
        :param user_guess: 猜測部分文本
        :param template_length: 模板長度
        :return: 填充後的 Prompt 或 None
        """
        surface_length = len(self.tokenizer.encode(surface, add_special_tokens=False))
        bottom_length = len(self.tokenizer.encode(bottom, add_special_tokens=False))
        user_guess_length = len(self.tokenizer.encode(user_guess, add_special_tokens=False))

        total_data_length = surface_length + bottom_length + user_guess_length

        for prompt in ["long", "medium", "short"]:
            prompt_length = self.prompts[prompt]["length"]
            if total_data_length  + prompt_length + template_length <= self.max_length - 2:  # 加上頭尾的[CLS]、[SEP]
                return self.prompts[prompt]["prompt"].format(surface=surface, bottom=bottom) + user_guess
        return None

    def _get_mask_index(self, input_ids):
        """獲取 [MASK] 的位置"""
        return (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    def __len__(self):
        """返回資料長度"""
        return len(self.data)

    def __getitem__(self, idx):
        """處理每筆資料"""
        item = self.data[idx]
        input_text = item["input_text"]
        label_id = torch.tensor(item["label_id"], dtype=torch.long) 

        # Tokenize 輸入和標籤
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        label_ids = torch.full(inputs["input_ids"].shape, -100)
        flags = torch.full(inputs["input_ids"].shape, 0)  # 1: [MASK], 2: template, 0: 其他

        mask_idx = self._get_mask_index(inputs["input_ids"])
        # 填充 [MASK] 的位置
        label_ids[0, mask_idx] = label_id

        # 建立 frag array
        template_ids_length = len(self.template.split(self.tokenizer._mask_token)[0])

        flags[0, mask_idx] = 1
        flags[0, mask_idx - template_ids_length:mask_idx] = 2

        # 輸出包含 input_ids, attention_mask 和 label_id
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_ids": label_ids.squeeze(0),
            "flags": flags.squeeze(0)
        }


class TurtleSoupClassificationDataset(Dataset):
    def __init__(self, data_path, prompt_path, tokenizer, template, label_map, max_length=512):
        """
        初始化 Dataset
        :param data_path: JSON 檔案路徑
        :param prompt_path: Prompt 檔案路徑
        :param template: 模板字符串
        :param label_map: 標籤映射字典
        :param max_length: 最大序列長度
        """
        self.data_path = Path(data_path)
        self.prompt_path = Path(prompt_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.template = template
        self.label_map = label_map

        self.prompts = self._load_prompts()
        self.data = self._load_data()

    def _load_prompts(self):
        """
        載入 Prompt 檔案並 Tokenize 計算長度
        :return: 包含 Tokenized 長度的 Prompt 字典
        """
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        tokenized_prompts = {}
        for key, value in prompts.items():
            prompt_text = value["prompt"].format(surface="", bottom="")
            tokenized_length = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

            tokenized_prompts[key] = {
                "prompt": value["prompt"],
                "length": tokenized_length
            }
        return tokenized_prompts

    def _load_data(self):
        """
        載入 JSON 檔案，選擇適合的 Prompt，並生成處理後的輸入文本與標籤。
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_data = []
        for entry in data:
            surface = entry["surface"]
            bottom = entry["bottom"]
            user_guess = entry["user_guess"]
            label = self._map_label(entry["label"])

            # 選擇適合的 prompt
            prompt_filled = self._select_prompt(surface, bottom, user_guess)
            if not prompt_filled:
                continue

            # 組合最終輸入文本
            input_text = prompt_filled + " " + user_guess

            # 加入處理後的數據
            processed_data.append({"input_text": input_text, "label": label})

        return processed_data

    def _map_label(self, label):
        """映射標籤為整數標籤"""
        return self.label_map[label]

    def _select_prompt(self, surface, bottom, user_guess):
        """
        根據數據長度選擇適合的 Prompt。
        """
        surface_length = len(self.tokenizer.encode(surface, add_special_tokens=False))
        bottom_length = len(self.tokenizer.encode(bottom, add_special_tokens=False))
        user_guess_length = len(self.tokenizer.encode(user_guess, add_special_tokens=False))

        total_data_length = surface_length + bottom_length + user_guess_length

        for prompt in ["long", "medium", "short"]:
            prompt_length = self.prompts[prompt]["length"]
            if total_data_length + prompt_length <= self.max_length - 2:  # [CLS] 和 [SEP]
                return self.prompts[prompt]["prompt"].format(surface=surface, bottom=bottom)
        return None

    def __len__(self):
        """返回資料長度"""
        return len(self.data)

    def __getitem__(self, idx):
        """處理每筆資料"""
        item = self.data[idx]
        input_text = item["input_text"]
        label = torch.tensor(item["label"], dtype=torch.long)

        # Tokenize 輸入文本
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": label
        }
