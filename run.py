import torch
from tqdm import tqdm
import argparse
from transformers import BertForMaskedLM, BertTokenizerFast, BertForSequenceClassification
import torch
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from src.dataset import ContrastiveLearningDataset, TurtleSoupDataset, TurtleSoupClassificationDataset
from src.utils import save_training_results, load_config
from src.model import ContrastiveTrainer, DiffPET, PET
from sklearn.model_selection import train_test_split
import os


def train_pet_model(config):
    """
    通用的 PET 訓練與驗證方法。
    """
    training_function = config.training_function
    pretrained_model_path = config.pretrained_model_path
    device = config.device
    train_data_path = config.train_data_path
    test_data_path = config.test_data_path
    prompt_path = config.prompt_path
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    save_results = config.save_results
    save_params = config.save_params

    template = config.template
    label_map = config.pet.label_map
    max_length = config.pet.max_length

    if config.pet.enable_contrastive_learning:
        contrastive_margin = config.pet.contrastive_margin
        lambda_weight = config.pet.lambda_weight

    model = BertForMaskedLM.from_pretrained(pretrained_model_path).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path)
    
    train_dataset = TurtleSoupDataset(train_data_path, prompt_path, tokenizer, max_length=max_length, template=template, label_map=label_map
    , enable_contrastive_learning=config.pet.enable_contrastive_learning)
    test_dataset = TurtleSoupDataset(test_data_path, prompt_path, tokenizer, max_length=max_length, template=template, label_map=label_map, enable_contrastive_learning=config.pet.enable_contrastive_learning)

    # 創建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=num_training_steps)

    if training_function == 'diffpet':
        labels = [value for key, value in label_map.items()]
        pet = DiffPET(model, tokenizer, template, labels, device)
    else:
        pet = PET(model, tokenizer, device)

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
            logits, total_loss = pet.forward_step(batch)

            # 計算準確率
            predicted_ids, true_ids = pet.get_predictions(batch, logits)
            correct_predictions += (predicted_ids == true_ids).sum().item()
            total_predictions += len(batch["input_ids"])  # Batch size

            # 對比學習損失
            contrastive_loss = 0.0
            if batch["contrastive"]:
                contrastive_loss = pet.compute_contrastive_loss(batch["contrastive"], contrastive_margin)
                total_loss += lambda_weight * contrastive_loss

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

    output_dir = config.output_dir
    if save_results:
        save_training_results(training_function, train_losses, train_accuracies, val_losses, val_accuracies, output_dir)

    if save_params:
        pet.save_model(os.path.join(output_dir, training_function))


def train_classification_model(config):
    pretrained_model_path = config.pretrained_model_path
    device = config.device
    train_data_path = config.train_data_path
    test_data_path = config.test_data_path
    prompt_path = config.prompt_path
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    save_results = config.save_results
    save_params = config.save_params


    max_length = config.classification.max_length
    label_map = config.classification.label_map

    model = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=3)
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path)

    train_dataset = TurtleSoupClassificationDataset(train_data_path, prompt_path, tokenizer, max_length=max_length, label_map=label_map)
    test_dataset = TurtleSoupClassificationDataset(test_data_path, prompt_path, tokenizer, max_length=max_length, label_map=label_map)

    # 創建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

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

    output_dir = config.output_dir
    if save_results:
        save_training_results('classification', train_losses, train_accuracies, val_losses, val_accuracies, output_dir)

    if save_params:
        model.save_pretrained(os.path.join(output_dir, "classification"))
        tokenizer.save_pretrained(os.path.join(output_dir, "classification"))

def train_contrastive_model(config):
    pretrained_model_path = config.pretrained_model_path
    device = config.device
    train_data_path = config.train_data_path
    prompt_path = config.prompt_path
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    save_results = config.save_results
    save_params = config.save_params

    max_length = config.contrastive.max_length
    contrastive_margin = config.contrastive.contrastive_margin

    model = BertForMaskedLM.from_pretrained(pretrained_model_path).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_path)
    
    contrastive_dataset = ContrastiveLearningDataset(
        train_data_path, prompt_path, tokenizer,
        max_length=max_length
    )

    train_indices, val_indices = train_test_split(
        list(range(len(contrastive_dataset))), test_size=0.15
    )
    train_dataset = torch.utils.data.Subset(contrastive_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(contrastive_dataset, val_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_training_steps = len(train_dataloader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=num_training_steps)

    trainer = ContrastiveTrainer(model, tokenizer, device)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        trainer.model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            # 計算對比學習損失
            loss = trainer.compute_contrastive_loss(batch, contrastive_margin)
            train_loss += loss.item()

            # 反向傳播與參數更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # 計算平均損失
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # 驗證階段
        trainer.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                loss = trainer.compute_contrastive_loss(batch, contrastive_margin)
                val_loss += loss.item()

        # 計算平均驗證損失
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

    output_dir = config.output_dir
    if save_results:
        save_training_results('contrastive', train_losses, [], val_losses, [], output_dir)

    if save_params:
        trainer.save_model(os.path.join(output_dir, "contrastive"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, 
        default="config/config_zh.yaml", 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--language", type=str, 
        choices=["zh", "en"], 
        required=True, 
        help="Language to train the model."
    )
    parser.add_argument(
        "--training_function", type=str, 
        choices=["classification", "pet", "diffpet", "contrastive"], 
        required=True, 
        help="Training function to use."
    )
    parser.add_argument("--save_results", type=lambda x: x.lower() == "true", default=False, help="Save training results to a file.")
    parser.add_argument("--save_params", type=lambda x: x.lower() == "true", default=False, help="Save model parameters to a file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    args = parser.parse_args()

    language = args.language
    config_path = f"config/config_{language}.yaml"
    override_params = {k: v for k, v in vars(args).items() if v is not None and k != "language"}

    config = load_config(config_path, **override_params)

    if args.training_function == "classification":
        train_classification_model(config)
    elif args.training_function == "pet" or "diffpet":
        train_pet_model(config)
    elif args.training_function == "contrastive":
        train_contrastive_model(config)