import torch
from typing import List, Dict, Any

"""
此實現參考自 DART 的 GitHub 專案 (https://github.com/zjunlp/DART/tree/main)。
它提供了基於 PET 和 DiffPET 的訓練框架，支持可微分的 tokens 和 prompts。
"""

class Trainer:
    """
    通用訓練基礎類別，提供基本功能：
    - 模型保存
    - 嵌入提取
    """
    def __init__(self, model, tokenizer, device):
        """
        初始化 Trainer。

        Args:
            model: Transformer 模型實例，用於嵌入提取。
            tokenizer: 用於編碼的 tokenizer。
            device: 設備類型 (如 GPU 或 CPU)。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def save_model(self, path: str):
        """
        儲存模型參數。

        Args:
            path (str): 儲存模型的目錄。
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

class ContrastiveTrainer(Trainer):
    """
    專門處理對比學習的訓練類別。
    """
    def _get_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        獲取文本的嵌入向量，使用 [CLS] token 的隱藏狀態作為句子嵌入。

        Args:
            inputs (dict): 包含 `input_ids` 和 `attention_mask` 的字典。

        Returns:
            Tensor: [CLS] token 的嵌入向量，形狀為 (batch_size, hidden_size)。
        """
        if "input_ids" not in inputs or "attention_mask" not in inputs:
            raise ValueError("Inputs must include 'input_ids' and 'attention_mask'.")

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 模型前向傳播
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # 使用最後一層隱藏狀態的 [CLS] token 表示作為句子嵌入
        embeddings = outputs.hidden_states[-1][:, 0, :]  # (batch_size, hidden_size)
        return embeddings

    def compute_contrastive_loss(self, contrastive_data, margin: float = 1.0) -> torch.Tensor:
        """
        計算對比學習損失。

        Args:
            contrastive_data (dict): 包含文本嵌入對與標籤的字典。
            margin (float): 負樣本的距離邊界。
            
        Returns:
            Tensor: 對比損失。
        """
        emb_a = self._get_embeddings(contrastive_data["text_a_inputs"])
        emb_b = self._get_embeddings(contrastive_data["text_b_inputs"])
        labels = contrastive_data["contrastive_label"].to(self.device)

        # 計算歐幾里得距離
        distances = torch.nn.functional.pairwise_distance(emb_a, emb_b)

        # 計算對比損失
        loss = torch.mean(
            labels * distances.pow(2) +
            (1 - labels) * torch.relu(margin - distances).pow(2)
        )

        return loss


class PET(ContrastiveTrainer):
    """
    PET (Pattern-Exploiting Training) 基礎類別，用於基於模板和標籤的分類模型。
    """
    def forward_step(self, batch):
        """
        前向傳播計算 logits 和損失。

        Args:
            batch (dict): 包含 `input_ids`, `attention_mask`, 和 `label_ids` 的字典。

        Returns:
            Tuple[Tensor, Tensor]: logits 和損失。
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        label_ids = batch["label_ids"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)

        logits = outputs.logits
        loss = outputs.loss

        return logits, loss

    def get_predictions(self, batch, logits):
        """
        根據 logits 生成模型的預測結果。
        """
        label_ids = batch["label_ids"].to(self.device)
        flags = batch["flags"].to(self.device)

        mask_indices = (flags == 1)
    
        # 提取 [MASK] 對應的 logits
        mask_logits = logits[mask_indices, :]  # [num_masks, vocab_size]

        # 預測的標籤 ID
        predicted_ids = torch.argmax(mask_logits, dim=-1)  # [batch_size, num_masks_per_example]
        
        # 獲取對應的真實標籤
        true_ids = label_ids[mask_indices]  # [batch_size, num_masks_per_example]

        return predicted_ids, true_ids
    
    def compute_contrastive_loss(self, contrastive_data_list: List, margin: float = 1.0) -> torch.Tensor:
        """
        計算一個 list 的對比學習損失（覆寫父類方法）。

        Args:
            contrastive_data_list (list): 包含多個文本嵌入對與標籤的列表。
            margin (float): 負樣本的距離邊界。

        Returns:
            Tensor: 對比損失。
        """
        total_loss = 0.0
        for contrastive_data in contrastive_data_list:
            loss = super().compute_contrastive_loss(contrastive_data, margin)
            total_loss += loss

        return total_loss

    

class DiffPET(PET):
    """
    DiffPET (Differentiable Prompt-based PET) 擴展類別。
    - 支持可微分模板和標籤的初始化和替換。
    - 允許模型學習模板和標籤的嵌入表示。

    Args:
        template: 用於生成輸入的模板字符串。
        label: 標籤的文本表示。
        template_token_mapping: 模板中 token 與新索引的對應關係。
        label_token_mapping: 標籤中 token 與新索引的對應關係。
    """
    def __init__(self, model, tokenizer, template: str, labels: List, device):
        super().__init__(model, tokenizer, device)
        self.template = template
        self.labels = labels
        self.template_token_mapping = []  # 儲存模板中每個 token 與其新索引的對應關係
        self.label_token_mapping = {}  # 儲存標籤中每個 token 與其新索引的對應關係

        # 儲存新索引的張量表示
        self.template_ids = []

        self._initialize_tokens()
        self._initialize_embeddings()

    def _initialize_tokens(self):
        """
        初始化模板和標籤的 token ID。
        """
        curr_idx = 1  # 從詞彙表最後開始分配新的 token 索引

        # 處理模板中的固定文本部分
        for segment in self.template.split('[MASK]'):
            if segment.strip():  # 避免處理空白部分
                token_ids = self.tokenizer.encode(segment, add_special_tokens=False)  # 將文本轉換為 token IDs
                for token_id in token_ids:
                    self.template_token_mapping.append((token_id, curr_idx))  # 映射原始 token ID 到新索引
                    curr_idx += 1

        # 處理標籤映射中的文本
        for label in self.labels:
            label_ids = self.tokenizer.encode(label, add_special_tokens=False)[0]  # 將標籤文本轉換為 token IDs
            self.label_token_mapping[label_ids] = curr_idx
            curr_idx += 1  # 更新索引

        self.template_ids = torch.tensor([new_id for _, new_id in self.template_token_mapping], device=self.device).long()


    def _initialize_embeddings(self, copy=True):
        """
        初始化新 token 的嵌入表示。
        - 若 copy=True，將原始 token 的嵌入複製到新 token。
        - 若 copy=False，隨機初始化新 token 的嵌入。
        """
        embedding_weights = self.model.get_input_embeddings().weight.data
        if copy:
            # 將原始 token 的嵌入複製到新 token
            for old_id, new_id in self.template_token_mapping:
                embedding_weights[new_id] = embedding_weights[old_id]
            for old_id, new_id in self.label_token_mapping.items():
                embedding_weights[new_id] = embedding_weights[old_id]
        else:
            # 隨機初始化新的 token 嵌入
            for _, new_id in self.template_token_mapping:
                embedding_weights[new_id].uniform_(-0.1, 0.1)
            for _, new_id in self.label_token_mapping.items():
                embedding_weights[new_id].uniform_(-0.1, 0.1)


    def _prepare_input(self, batch):
        """
        替換輸入中的模板和標籤 token。
        """
        input_ids = batch["input_ids"].to(self.device)
        label_ids = batch["label_ids"].to(self.device)
        flags = batch["flags"]
        batch_size = input_ids.size(0)

        template_positions = (flags == 2)
        input_ids[template_positions] = self.template_ids.repeat(batch_size)

        mask_positions = (flags == 1)
        old_label_ids = label_ids[mask_positions]

        label_ids[mask_positions] = torch.tensor([self.label_token_mapping[id.item()] for id in old_label_ids], device=self.device)

        batch["input_ids"] = input_ids
        batch["label_ids"] = label_ids

    def forward_step(self, batch):
        """
        準備輸入後執行前向傳播。
        """
        self._prepare_input(batch)
        return super().forward_step(batch)
