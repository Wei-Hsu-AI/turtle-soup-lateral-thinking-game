import json
from torch.utils.data import Dataset
from pathlib import Path
import torch
from typing import Any

class ContrastiveLearningMixin:
    def _generate_contrastive_pairs(self, surface: str, bottom: str, user_guess: str, label: str) -> list[dict[str, Any]]:
        """
        根據標籤生成對比學習樣本對。
        """
        contrastive_pairs = []

        if label == 'T':
            contrastive_pairs.extend([
                {"text_a": surface, "text_b": user_guess, "label": 1},
                {"text_a": bottom, "text_b": user_guess, "label": 1},
            ])
        elif label == 'F':
            negative_sample = {"text_a": bottom, "text_b": user_guess, "label": 0}
            contrastive_pairs.extend([negative_sample] * 2)
        elif label == 'N':
            contrastive_pairs.extend([
                {"text_a": surface, "text_b": user_guess, "label": 0},
                {"text_a": bottom, "text_b": user_guess, "label": 0},
            ])

        return contrastive_pairs

    def _encode_contrastive_sample(self, text_a: str, text_b: str, label: int) -> dict[str, Any]:
        """
        將 text_a 和 text_b 進行編碼，並返回編碼後的數據結構。

        Args:
            text_a (str): 第一段文本。
            text_b (str): 第二段文本。
            label (int): 標籤值 (1 或 0)。

        Returns:
            Dict[str, Any]: 包含編碼後的數據結構。
        """
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("'ClassificationMixin' 需要 'tokenizer' 屬性來進行編碼。")

        if not hasattr(self, 'max_length'):
            raise AttributeError("'ClassificationMixin' 需要 'max_length' 屬性來限制序列長度。")

        text_a_encoded = self.tokenizer(
            text_a,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        text_b_encoded = self.tokenizer(
            text_b,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "text_a_encoded": {
                "input_ids": text_a_encoded["input_ids"].squeeze(0),
                "attention_mask": text_a_encoded["attention_mask"].squeeze(0),
            },
            "text_b_encoded": {
                "input_ids": text_b_encoded["input_ids"].squeeze(0),
                "attention_mask": text_b_encoded["attention_mask"].squeeze(0),
            },
            "contrastive_label": torch.tensor(label, dtype=torch.float)
        }



class ClassificationMixin:
    def _calculate_prompts_length(self, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
        """
        計算每個 Prompt 的長度，並返回包含文本和長度的列表。

        Args:
            prompts (list): Prompt 列表，每個 Prompt 包含 "text" 鍵對應模板字符串。

        Returns:
            list: 包含每個 Prompt 文本和長度的字典列表。
        """
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("'ClassificationMixin' 需要 'tokenizer' 屬性來進行編碼。")

        tokenized_prompts_list = []
        for prompt_item in prompts:
            prompt_text = prompt_item["text"].format(surface="", bottom="", user_guess="")
            prompt_length = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))

            tokenized_prompts_list.append({
                "text": prompt_text,
                "length": prompt_length
            })

        return tokenized_prompts_list

    def _processe_data(self, surface: str, bottom: str, user_guess: str, label: Any, prompts: list[dict[str, Any]]) -> dict[str, Any]:
        """
        根據提供的 surface, bottom, user_guess 和 label，選擇適合的 Prompt 並生成輸入數據。

        Args:
            surface (str): 表層文本。
            bottom (str): 底層文本。
            user_guess (str): 用戶猜測文本。
            label (Any): 標籤數據。
            prompts (list[dict[str, Any]]): 包含文本和長度的 Prompt 列表。

        Returns:
            dict[str, Any]: 包含 "input_text" 和 "label" 的字典，若無法選擇 Prompt 則返回 None。
        """
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("'ClassificationMixin' 需要 'tokenizer' 屬性來進行編碼。")

        if not hasattr(self, 'max_length'):
            raise AttributeError("'ClassificationMixin' 需要 'max_length' 屬性來限制序列長度。")

        total_data_length = len(self.tokenizer.encode(surface + bottom + user_guess, add_special_tokens=False))

        selected_prompt_text = self._select_prompt(prompts, total_data_length)
        # 如果 Prompt 與數據總長度超過限制，跳過該數據
        if not selected_prompt_text:
            return None

        formatted_prompt = selected_prompt_text.format(surface=surface, bottom=bottom, user_guess=user_guess)

        return {"input_text": formatted_prompt, "label": label}

    def _select_prompt(self, prompts: list[dict[str, Any]], total_data_length: int) -> str | None:
        """
        根據數據長度選擇適合的 Prompt。

        Args:
            prompts (list[dict[str, Any]]): 包含文本和長度的 Prompt 列表。
            total_data_length (int): 數據的編碼長度。

        Returns:
            str | None: 適合的 Prompt 文本，若無法選擇則返回 None。
        """
        if not hasattr(self, 'max_length'):
            raise AttributeError("'ClassificationMixin' 需要 'max_length' 屬性來限制序列長度。")

        for prompt_item in prompts:
            prompt_length = prompt_item["length"]
            if total_data_length + prompt_length <= self.max_length - 2:  # [CLS] 和 [SEP]
                return prompt_item["text"]
            
        return None


class TurtleSoupDataset(Dataset, ContrastiveLearningMixin, ClassificationMixin):
    def __init__(self, data_path: str, prompt_path: str, tokenizer, template: str, label_map: dict[str, str] = None, max_length: int = 512, enable_contrastive_learning: bool = False):
        """
        初始化 Dataset

        Args:
            data_path (str): JSON 檔案路徑。
            prompt_path (str): Prompt 檔案路徑。
            tokenizer: 用於文本編碼的 tokenizer 實例。
            template (str): 模板字符串。
            label_map (dict[str, str], optional): 標籤映射字典。默認為 None。
            max_length (int): 最大序列長度。
            contrastive_learning (bool): 是否啟用對比學習。默認為 False。
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
        self.template_ids_length = len(self.tokenizer.encode(template, add_special_tokens=False))

        self.label_map = label_map
        self.enable_contrastive_learning = enable_contrastive_learning

        self.prompts = self._load_prompts()
        self.data = self._load_data()

    def _load_prompts(self):
        """
        加載 Prompt 並附加模板字符串。
        """
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")
        
        for entry in prompts:
            entry['text'] += self.template

        return self._calculate_prompts_length(prompts)
        
    def _load_data(self):
        """
        加載數據並處理成可用於訓練或推理的格式。
        """
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

        processed_data_list = []
        for entry in data:
            if not all(key in entry for key in ("surface", "bottom", "user_guess", "label")):
                raise ValueError("數據條目缺少必要字段: surface, bottom, user_guess, label。")

            surface = entry["surface"]  # 湯面部分
            bottom = entry["bottom"]    # 湯底部分
            user_guess = entry["user_guess"]  # 玩家猜測
            label = self._map_label(entry["label"])  # 將標籤映射為可用的標籤名稱

            mlm_processed_data = self._processe_data(surface, bottom, user_guess, label, self.prompts)
            if not mlm_processed_data:
                print(f"Skipped entry due to length: {entry}")
                continue

            contrastive_pairs = None
            if self.enable_contrastive_learning:
                # 生成對比學習樣本對
                contrastive_pairs = self._generate_contrastive_pairs(surface, bottom, user_guess, entry["label"])

            # 加入處理後的數據
            processed_data_list.append({
                "mlm": mlm_processed_data,
                "contrastive": contrastive_pairs,
            })

        return processed_data_list
    
    def _map_label(self, label: str):
        """映射標籤為對應的中文描述"""
        if self.label_map is not None:
            label = self.label_map[label]
    
        return label
    
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
        根據索引處理並返回數據。
        """
        item = self.data[idx]

        # 處理 MLM 資料
        mlm_data = item['mlm']
        input_text = mlm_data["input_text"]
        label = mlm_data["label"]

        label_id = self.tokenizer.encode(
            label,
            add_special_tokens=False
        )[0]

        # Tokenize 輸入和標籤
        tokenized_inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        label_ids = torch.full(tokenized_inputs["input_ids"].shape, -100)
        flags = torch.full(tokenized_inputs["input_ids"].shape, 0)  # 1: [MASK], 2: template, 0: 其他

        # 找到 [MASK] 的索引並標記
        mask_idx = self._get_mask_index(tokenized_inputs["input_ids"])
        if mask_idx is None:
            raise(f"Warning: [MASK] not found in input_text: {input_text}")

        # 填充 [MASK] 的位置
        label_ids[0, mask_idx] = label_id
        # 建立 flags array
        template_ids_without_mask_length = self.template_ids_length - 1
        flags[0, mask_idx] = 1
        flags[0, mask_idx - template_ids_without_mask_length:mask_idx] = 2

        # 處理對比學習資料
        contrastive_pairs = item['contrastive']
        contrastive_sample_list = []
        if contrastive_pairs:
            for pair in contrastive_pairs:
                encoded_contrastive_sample = self._encode_contrastive_sample(
                    text_a=pair["text_a"],
                    text_b=pair["text_b"],
                    label=pair["label"],
                )
                contrastive_sample_list.append(encoded_contrastive_sample)

        # 返回處理後的數據
        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "label_ids": label_ids.squeeze(0),
            "flags": flags.squeeze(0),
            "contrastive": contrastive_sample_list
        }
    

class ContrastiveLearningDataset(Dataset, ContrastiveLearningMixin):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
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

        contrastive_pairs_list = []
        for entry in data:
            if not all(key in entry for key in ("surface", "bottom", "user_guess", "label")):
                raise ValueError("數據條目缺少必要字段: surface, bottom, user_guess, label。")
            
            surface = entry["surface"]
            bottom = entry["bottom"]
            user_guess = entry["user_guess"]
            label = entry["label"]

            # 生成對比學習樣本對
            contrastive_pairs = self._generate_contrastive_pairs(surface, bottom, user_guess, label)
            contrastive_pairs_list.extend(contrastive_pairs)

        return contrastive_pairs_list

    def __len__(self):
        """返回數據長度。"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回處理後的對比學習樣本。
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
        )


class TurtleSoupClassificationDataset(Dataset, ClassificationMixin):
    def __init__(self, data_path: str, prompt_path: str, tokenizer, label_map: dict[str, int], max_length: int = 512):
        """
        初始化 Dataset

        Args:
            dataset_path (str): JSON 數據文件路徑。
            prompt_path (str): Prompt 文件路徑。
            label_mapping (dict[str, int]): 標籤映射字典。
            max_sequence_length (int): 最大序列長度。
        """
        self.data_path = Path(data_path)
        self.prompt_path = Path(prompt_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {data_path}")
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"指定的文件不存在: {prompt_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.label_map = label_map

        self.prompts = self._load_prompts()
        self.data = self._load_data()

    def _load_prompts(self):
        """
        加載 Prompt 文件並計算每個 Prompt 的長度。
        """
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompts = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

        return self._calculate_prompts_length(prompts)
    
    def _load_data(self):
        """
        加載數據並處理成可用於訓練或推理的格式。
        """
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"無法解析 JSON 文件: {e}")

        processed_data_list = []
        for entry in data:
            if not all(key in entry for key in ("surface", "bottom", "user_guess", "label")):
                raise ValueError("數據條目缺少必要字段: surface, bottom, user_guess, label。")

            surface = entry["surface"]  # 湯面部分
            bottom = entry["bottom"]    # 湯底部分
            user_guess = entry["user_guess"]  # 玩家猜測
            label = self._map_label(entry["label"])  # 將標籤映射為可用的標籤名稱

            mlm_processed_data = self._processe_data(surface, bottom, user_guess, label, self.prompts)
            if not mlm_processed_data:
                print(f"Skipped entry due to length: {entry}")
                continue

            processed_data_list.append(mlm_processed_data)

        return processed_data_list

    def _map_label(self, label: str):
        """映射標籤為對應的類別"""
        if self.label_map is not None:
            label = self.label_map[label]
    
        return label

    def __len__(self):
        """返回資料長度"""
        return len(self.data)

    def __getitem__(self, idx):
        """處理每筆資料"""
        item = self.data[idx]
        input_text = item["input_text"]
        label = torch.tensor(item["label"], dtype=torch.long)

        # Tokenize 輸入文本
        tokenized_inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized_inputs["input_ids"].squeeze(0),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze(0),
            "label": label
        }
