import torch
from tqdm import tqdm


def train_pet_model(pet, train_dataloader, val_dataloader, optimizer, lr_scheduler, epochs, λ=0.5, contrastive_margin=1.0):
    """
    通用的 PET 訓練與驗證方法。

    Args:
        pet: PET 或 DiffPET 模型實例。
        train_dataloader: 訓練數據加載器。
        val_dataloader: 驗證數據加載器。
        optimizer: 用於參數更新的優化器。
        lr_scheduler: 學習率調度器。
        epochs: 總訓練輪數。
        λ (float): 控制分類損失與對比損失的權重比例。
        contrastive_margin (float): 對比學習中負樣本的距離邊界。

    Returns:
        train_losses: 每個 epoch 的訓練損失。
        train_accuracies: 每個 epoch 的訓練準確率。
        val_losses: 每個 epoch 的驗證損失。
        val_accuracies: 每個 epoch 的驗證準確率。
    """
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # 訓練階段
        pet.model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            # 前向傳播
            logits, loss_cls = pet.forward_step(batch)

            # 計算準確率
            predicted_ids, true_ids = pet.get_predictions(batch, logits)
            correct_predictions += (predicted_ids == true_ids).sum().item()
            total_predictions += len(batch["input_ids"])  # Batch size

            # 對比學習損失
            contrastive_loss = 0.0
            if batch["contrastive"]:
                contrastive_loss = pet.compute_contrastive_loss(batch["contrastive"], contrastive_margin)

            # 合併損失
            total_loss = loss_cls + λ * contrastive_loss
            train_loss += total_loss.item()

            # 反向傳播與參數更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # 計算訓練指標
        train_loss /= len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 驗證階段（只計算分類性能）
        pet.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_dataloader:
                logits, loss = pet.forward_step(batch)

                predicted_ids, true_ids = pet.get_predictions(batch, logits)
                correct_predictions += (predicted_ids == true_ids).sum().item()
                total_predictions += len(batch["input_ids"])
                
                val_loss += loss.item()

        # 計算驗證指標
        val_loss /= len(val_dataloader)
        val_accuracy = correct_predictions / total_predictions
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, train_acc: {train_accuracy:.3f}, val_acc: {val_accuracy:.3f}")

    return train_losses, train_accuracies, val_losses, val_accuracies


def train_classification_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, epochs, device):
    """
    通用的文本分類訓練與驗證方法。

    功能：
    - 執行模型的訓練迴圈，包括前向傳播、反向傳播以及參數更新。
    - 提供訓練與驗證的準確率與損失記錄。

    參數：
    - model: BertForSequenceClassification 模型實例。
    - train_dataloader: 訓練數據加載器。
    - val_dataloader: 驗證數據加載器。
    - optimizer: 用於參數更新的優化器。
    - lr_scheduler: 學習率調度器。
    - epochs: 總訓練輪數。
    - device: 設備類型 (如 GPU 或 CPU)。

    返回：
    - train_losses: 每個 epoch 的訓練損失。
    - train_accuracies: 每個 epoch 的訓練準確率。
    - val_losses: 每個 epoch 的驗證損失。
    - val_accuracies: 每個 epoch 的驗證準確率。
    """
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    model.to(device)

    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向傳播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            train_loss += loss.item()

            # 計算準確率
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # 反向傳播與參數更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # 計算訓練指標
        train_loss /= len(train_dataloader)
        train_accuracy = correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 驗證階段
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()

                # 計算準確率
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        # 計算驗證指標
        val_loss /= len(val_dataloader)
        val_accuracy = correct_predictions / total_predictions
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, "
              f"train_acc: {train_accuracy:.3f}, val_acc: {val_accuracy:.3f}")

    return train_losses, train_accuracies, val_losses, val_accuracies