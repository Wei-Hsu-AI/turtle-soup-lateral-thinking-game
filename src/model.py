import torch

"""
此實現參考自 DART 的 GitHub 專案 (https://github.com/zjunlp/DART/tree/main)。
它提供了基於 PET 和 DiffPET 的訓練框架，支持可微分的 tokens 和 prompts。
"""

class PET:
    """
    PET (Pattern-Exploiting Training) 基礎類別，用於基於模板和標籤的分類模型。

    功能：
    - 支持模型的前向傳播和損失計算。
    - 提供預測方法來生成模型輸出。

    屬性：
    - model: 用於訓練的 Transformer 模型。
    - tokenizer: 用於編碼的 tokenizer。
    - device: 設備類型 (如 GPU 或 CPU)。
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward_step(self, batch):
        """
        前向傳播計算 logits 和損失。
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
    
    def save_model(self, path):
        """
        儲存模型參數
        """
        self.model.save_pretrained(path)
    

class DiffPET(PET):
    """
    DiffPET (Differentiable Prompt-based PET) 擴展類別。

    功能：
    - 支持可微分模板和標籤的初始化和替換。
    - 允許模型學習模板和標籤的嵌入表示。

    屬性：
    - template: 用於生成輸入的模板字符串。
    - label: 標籤的文本表示。
    - template_token_mapping: 模板中 token 與新索引的對應關係。
    - label_token_mapping: 標籤中 token 與新索引的對應關係。
    """
    def __init__(self, model, tokenizer, template, labels, device):
        super().__init__(model, tokenizer, device)
        self.template = template
        self.labels = labels
        self.template_token_mapping = []  # 儲存模板中每個 token 與其新索引的對應關係
        self.label_token_mapping = {}  # 儲存標籤中每個 token 與其新索引的對應關係

        # 儲存新索引的張量表示
        self.template_ids = []

        self._initialize_tokens()
        self._initialize_embeddings()

    # def _initialize_tokens(self):
    #     """
    #     初始化模板和標籤的 token ID。
    #     """
    #     vocab_size = self.tokenizer.vocab_size
    #     curr_idx = vocab_size - 1  # 從詞彙表最後開始分配新的 token 索引

    #     # 處理模板中的固定文本部分
    #     for segment in self.template.split(self.tokenizer._mask_token):
    #         if segment.strip():  # 避免處理空白部分
    #             token_ids = self.tokenizer.encode(segment, add_special_tokens=False)  # 將文本轉換為 token IDs
    #             for token_id in token_ids:
    #                 self.template_token_mapping.append((token_id, curr_idx))  # 映射原始 token ID 到新索引
    #                 curr_idx -= 1

    #     # 處理標籤映射中的文本
    #     for label in self.labels:
    #         label_ids = self.tokenizer.encode(label, add_special_tokens=False)[0]  # 將標籤文本轉換為 token IDs
    #         self.label_token_mapping[label_ids] = curr_idx
    #         curr_idx -= 1  # 更新索引

    #     self.template_ids = torch.tensor([new_id for _, new_id in self.template_token_mapping], device=self.device).long()

    def _initialize_tokens(self):
        """
        初始化模板和標籤的 token ID。
        """
        vocab_size = self.tokenizer.vocab_size
        curr_idx = 1  # 從詞彙表最後開始分配新的 token 索引

        # 處理模板中的固定文本部分
        for segment in self.template.split(self.tokenizer._mask_token):
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
    
    def decode_output(self, output_ids):
        reverse_label_token_mapping = {v: k for k, v in self.label_token_mapping.items()}

        """
        將模型輸出的新 token 轉換回原始 token。
        """
        decoded_tokens = []
        for token_id in output_ids:
            if token_id in reverse_label_token_mapping:
                # 標籤 token
                decoded_tokens.append(reverse_label_token_mapping[token_id])
            else:
                # 其他 token
                decoded_tokens.append(self.tokenizer.decode([token_id]))
        return decoded_tokens
    
    def batch_decode_output(self, batch_output_ids):
        """
        將批量輸出的新 token 轉換回原始 token。
        """
        decoded_batches = []
        for output_ids in batch_output_ids:
            decoded_batches.append(self.decode_output(output_ids.tolist()))
        return decoded_batches


