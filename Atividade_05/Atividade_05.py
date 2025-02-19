import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#Carrega o tokenizer e o modelo GPT-2 pre-treinado
meu_modelo = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(meu_modelo)
tokenizer.pad_token = tokenizer.eos_token 
model = GPT2LMHeadModel.from_pretrained(meu_modelo)

class MyCustomModel(nn.Module):
    def __init__(self, gpt2_model):
        super(MyCustomModel, self).__init__()
        self.gpt2 = gpt2_model
        #Adicione outras camadas ou modificações aqui

    def forward(self, input_ids, attention_mask=None):
        #Use o GPT-2 como parte do sei=u modelo
        outputs = self.gpt2(input_ids = input_ids, attention_mask = attention_mask)
        return outputs    
    def generate(self, input_ids, attention_mask=None):
        return self.gpt2.generate(input_ids, attention_mask)    

my_model = MyCustomModel(model)

num_epochs = 3
optimizer = torch.optim.AdamW(my_model.parameters(), lr=5e-5)

# Carrega um conjunto de dados de exemplo (usando o dataset "wikitext" da Hugging Face)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Função para tokenizar o conjunto de dados
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Aplica a tokenização ao conjunto de dados
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Converte o conjunto de dados tokenizado em um DataLoader
# Aqui, precisamos empacotar os tensores manualmente
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]  
    attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]  
    return {
        "input_ids": torch.stack(input_ids),  
        "attention_mask": torch.stack(attention_mask),  
    }

# Converte o conjunto de dados tokenizado em um DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model = MyCustomModel(model).to(device)

# Função para gerar a próxima palavra
def gerar_proxima_palavra(model, tokenizer, prompt, max_length=50):
    # Tokeniza o prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Gera a próxima palavra (ou sequência de palavras)
    output = model.gpt2.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2,  
        do_sample=True,  
        top_k=50, 
        top_p=0.95,  
    )

    # Decodifica a saída para texto
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Exemplo de uso
prompt = "No Brasil as pessoas gostam de"  
generated_text = gerar_proxima_palavra(my_model, tokenizer, prompt)
print("Texto gerado:", generated_text)