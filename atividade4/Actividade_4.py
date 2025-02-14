import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import torch.nn as nn

# Função para carregar o dataset
def read_data_set(file_path):
    try:
        df = pd.read_csv(file_path)
        return df 
    except Exception as e: 
        print(f"Erro ao carregar o dataset: {e}")
    print(f"Dataset carregado com sucesso: {file_path}")
    return None

# Função para dividir o dataset
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    print(f"Treino: {len(train_df)}, Validação: {len(val_df)}, Teste: {len(test_df)}")
    return train_df, val_df, test_df

# Classe CustomDataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

# Função collate_fn
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Função para calcular a acurácia
def compute_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    return (preds == labels).float().mean()

# Função de avaliação
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_accuracy += compute_accuracy(logits, labels).item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_val_loss = total_loss / len(val_loader)
    avg_val_accuracy = total_accuracy / len(val_loader)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    return avg_val_loss, avg_val_accuracy, f1_micro, f1_macro, all_labels, all_preds

# Função de treinamento
def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0
        total_accuracy = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += compute_accuracy(logits, labels).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")

        avg_val_loss, avg_val_accuracy, f1_micro, f1_macro, all_labels, all_preds = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")
        
        return f1_micro, f1_macro, all_labels, all_preds

# Função principal
def process_datasets(file_path):
    df = read_data_set(file_path)
    if df is None:
        return None
    
    train_df, val_df, test_df = split_data(df)

    # Mapeamento de classes
    class_mapping = {label: idx for idx, label in enumerate(train_df['class'].unique())}
    print(f"Class Mapping: {class_mapping}")

    # Aplicar mapeamento
    train_df['class'] = train_df['class'].map(class_mapping)
    val_df['class'] = val_df['class'].map(class_mapping)
    test_df['class'] = test_df['class'].map(class_mapping)

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Datasets
    train_dataset = CustomDataset(train_df['text'].tolist(), train_df['class'].tolist(), tokenizer)
    val_dataset = CustomDataset(val_df['text'].tolist(), val_df['class'].tolist(), tokenizer)
    test_dataset = CustomDataset(test_df['text'].tolist(), test_df['class'].tolist(), tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Modelo
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(class_mapping))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Otimizador e função de perda
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Treinamento
    return train(model, train_loader, val_loader, optimizer, criterion, device, epochs=3)

# Processar os datasets
file_path1 = 'Dmoz-Sports.csv'
file_path2 = 'SyskillWebert.csv'

print("\nProcessando Dmoz-Sports:")
f1_micro_1, f1_macro_1, labels_1, preds_1 = process_datasets(file_path1)

print("\nProcessando SyskillWebert:")
f1_micro_2, f1_macro_2, labels_2, preds_2 = process_datasets(file_path2)
