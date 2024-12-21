import json
from torch.utils.data import Dataset
from pathlib import Path
import torch
from typing import List, Dict, Any

class ContrastiveLearningMixin:
    def _generate_contrastive_pairs(self, surface: str, bottom: str, user_guess: str, label: str) -> List[Dict[str, Any]]:
        """
        根據標籤生成對比學習樣本對。
        """
        contrastive_pair = []

        if label == 'T':
            contrastive_pair.extend([
                {"text_a": surface, "text_b": user_guess, "label": 1},
                {"text_a": bottom, "text_b": user_guess, "label": 1},
            ])
        elif label == 'F':
            negative_sample = {"text_a": bottom, "text_b": user_guess, "label": 0}
            contrastive_pair.extend([negative_sample] * 2)
        elif label == 'N':
            contrastive_pair.extend([
                {"text_a": surface, "text_b": user_guess, "label": 0},
                {"text_a": bottom, "text_b": user_guess, "label": 0},
            ])

        return contrastive_pair

    def _encode_contrastive_sample(self, text_a: str, text_b: str, label: int, tokenizer, max_length: int) -> Dict[str, Any]:
        """
        將 text_a 和 text_b 進行編碼，並返回編碼後的數據結構。

        Args:
            text_a (str): 第一段文本。
            text_b (str): 第二段文本。
            label (int): 標籤值 (1 或 0)。
            tokenizer: 負責編碼的 tokenizer。
            max_length (int): 最大序列長度。

        Returns:
            Dict[str, Any]: 包含編碼後的數據結構。
        """
        text_a_inputs = tokenizer(
            text_a,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        text_b_inputs = tokenizer(
            text_b,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {
            "text_a_inputs": {
                "input_ids": text_a_inputs["input_ids"].squeeze(0),
                "attention_mask": text_a_inputs["attention_mask"].squeeze(0),
            },
            "text_b_inputs": {
                "input_ids": text_b_inputs["input_ids"].squeeze(0),
                "attention_mask": text_b_inputs["attention_mask"].squeeze(0),
            },
            "contrastive_label": torch.tensor(label, dtype=torch.float)
        }


class TurtleSoupDataset(Dataset, ContrastiveLearningMixin):
    def __init__(self, data_path: str, prompt_path: str, tokenizer, template: str, label_map: Dict[str, str], max_length: int = 512, contrastive_learning: bool = False):
        """
        初始化 Dataset

        Args:
            data_path: JSON 檔案路徑
            prompt_path: Prompt 檔案路徑
            template: 模板字符串
            label_map: 標籤映射字典
            max_length: 最大序列長度
            contrastive_learning: 是否要進行對比學習
        """
        self.data_path = Path(data_path)
        self.prompt_path = Path(prompt_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {data_path}")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {prompt_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.template = template
        self.label_map = label_map
        self.contrastive_learning = contrastive_learning

        self.prompts = self._load_prompts()
        self.data = self._load_data()


    def _load_prompts(self):
        """
        載入 Prompt 檔案並 Tokenize 計算長度
        
        Returns:
            包含 Tokenized 長度的 Prompt 字典
        """
        # 載入 Prompt 檔案，包含 short, medium, long Prompt 的結構
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

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
        載入 JSON 檔案，根據數據的長度動態選擇適合的 Prompt，
        並生成處理後的輸入文本與標籤數據，根據需要生成對比學習樣本。
        
        Returns:
            List[dict]: 處理後的數據列表，每筆數據包含：
                - `mlm`:
                    - `input_text` (str): 經過填充的最終輸入文本。
                    - `label` (int): 詞彙 ID 標籤。
                - `contrastive` (List[dict] or None): 
                    如果啟用對比學習，則為生成的樣本對，否則為 None。
        """
        # 載入數據檔案 (包含 surface, bottom, user_guess, label)
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

        self.template_ids_length = len(self.tokenizer.encode(self.template, add_special_tokens=False))

        processed_data = []
        for entry in data:
            if not all(key in entry for key in ("surface", "bottom", "user_guess", "label")):
                raise ValueError("數據條目缺少必要字段: surface, bottom, user_guess, label。")

            surface = entry["surface"]  # 湯面部分
            bottom = entry["bottom"]    # 湯底部分
            user_guess = entry["user_guess"]  # 玩家猜測
            label = self._map_label(entry["label"])  # 將標籤映射為可用的標籤名稱

            # 根據模板選擇適合的 Prompt
            prompt_filled = self._select_prompt(surface, bottom, user_guess, self.template_ids_length)

            # 如果 Prompt 與數據總長度超過限制，跳過該數據
            if not prompt_filled:
                print(f"Skipped entry due to length: {entry}")
                continue

            # 合成最終的輸入文本，包含填充後 Prompt、玩家猜測和模板
            input_text = prompt_filled + self.template

            # 將 label 轉換為詞彙 ID
            label_id = self.tokenizer.encode(
                label,
                add_special_tokens=False
            )[0]

            contrastive_pair = None
            if self.contrastive_learning:
                # 生成對比學習樣本對
                contrastive_pair = self._generate_contrastive_pairs(surface, bottom, user_guess, entry["label"])

            # 加入處理後的數據
            processed_data.append({
                "mlm": {"input_text": input_text, "label_id": label_id},
                "contrastive": contrastive_pair,
            })

        return processed_data
    
    def _map_label(self, label: str):
        """映射標籤為對應的中文描述"""
        return self.label_map[label]
    
    def _select_prompt(self, surface: str, bottom: str, user_guess: str, template_length: int):
        """
        根據數據長度選擇適合的 Prompt

        Returns: 
            填充後的 Prompt 或 None
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
        mask_index = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if mask_index.numel() == 0:
            raise ValueError("未找到 [MASK] 標記位置。")
        return mask_index

    def __len__(self):
        """返回資料長度"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        處理每筆資料，生成適合模型的輸入格式。

        Args:
            idx (int): 資料的索引。

        Returns:
            dict: 包含處理後的 MLM 和對比學習數據。
        """
        item = self.data[idx]

        # 處理 MLM 資料
        mlm_data = item['mlm']
        input_text = mlm_data["input_text"]
        label_id = torch.tensor(mlm_data["label_id"], dtype=torch.long) 

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

        # 找到 [MASK] 的索引並標記
        mask_idx = self._get_mask_index(inputs["input_ids"])
        if mask_idx is None:
            raise(f"Warning: [MASK] not found in input_text: {input_text}")

        # 填充 [MASK] 的位置
        label_ids[0, mask_idx] = label_id
        # 建立 flags array
        template_ids_without_mask_length = self.template_ids_length - 1
        flags[0, mask_idx] = 1
        flags[0, mask_idx - template_ids_without_mask_length:mask_idx] = 2

        # 處理對比學習資料
        contrastive_data = item['contrastive']
        contrastive_pairs = []
        if contrastive_data:
            for pair in contrastive_data:
                contrastive_sample = self._encode_contrastive_sample(
                    text_a=pair["text_a"],
                    text_b=pair["text_b"],
                    label=pair["label"],
                    tokenizer=self.tokenizer,
                    max_length=self.max_length
                )
                contrastive_pairs.append(contrastive_sample)

        # 返回處理後的數據
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_ids": label_ids.squeeze(0),
            "flags": flags.squeeze(0),
            "contrastive": contrastive_pairs
        }
    

class ContrastiveLearningDataset(Dataset, ContrastiveLearningMixin):
    def __init__(self, data_path, tokenizer, max_length:int = 512):
        """
        專用於對比學習的 Dataset。
        
        Args:
            data_path (str): JSON 數據文件路徑。
            tokenizer: 用於文本編碼的 tokenizer。
            max_length (int): 最大序列長度。
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {data_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 載入並處理數據
        self.data = self._load_data()

    def _load_data(self):
        """
        載入數據並處理對比學習樣本對。
        
        Returns:
            List[dict]: 每筆數據包含 text_a, text_b 和對應的標籤。
        """
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

        processed_data = []
        for entry in data:
            if not all(key in entry for key in ("surface", "bottom", "user_guess", "label")):
                raise ValueError("數據條目缺少必要字段: surface, bottom, user_guess, label。")
            
            surface = entry["surface"]
            bottom = entry["bottom"]
            user_guess = entry["user_guess"]
            label = entry["label"]

            # 生成對比學習樣本對
            pairs = self._generate_contrastive_pairs(surface, bottom, user_guess, label)
            processed_data.extend(pairs)

        return processed_data

    def __len__(self):
        """返回數據長度。"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回處理後的對比學習樣本。

        Args:
            idx (int): 樣本索引。

        Returns:
            dict: 包含對比學習樣本的輸入格式和標籤。
        """
        item = self.data[idx]
        text_a = item["text_a"]
        text_b = item["text_b"]
        label = item["label"]

        # 使用 Mixin 中的方法進行編碼
        return self._encode_contrastive_sample(
            text_a=text_a,
            text_b=text_b,
            label=label,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )



class TurtleSoupClassificationDataset(Dataset):
    def __init__(self, data_path: str, prompt_path: str, tokenizer, template: str, label_map: Dict[str, int], max_length:int = 512):
        """
        初始化 Dataset

        Args:
            data_path: JSON 檔案路徑
            prompt_path: Prompt 檔案路徑
            template: 模板字符串
            label_map: 標籤映射字典
            max_length: 最大序列長度
        """
        self.data_path = Path(data_path)
        self.prompt_path = Path(prompt_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {data_path}")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {prompt_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.template = template
        self.label_map = label_map

        self.prompts = self._load_prompts()
        self.data = self._load_data()

    def _load_prompts(self):
        """
        載入 Prompt 檔案並 Tokenize 計算長度

        Returns:
            包含 Tokenized 長度的 Prompt 字典
        """
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

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
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

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
