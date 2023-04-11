import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from MultiClassDataset import MultiClassDataset
from konlpy.tag import Okt
okt = Okt()

# 경로 설정
train_set_path = 'data/train_set.xlsx'
model_save_path = 'trained_models/test.pth'

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 1
learning_rate = 2e-5

# 학습 데이터셋 불러오기
data_set = pd.read_excel(train_set_path)

# 본문 명사 추출
for i in range(len(data_set['text'])):
    data_set.loc[i, 'text'] = " ".join(okt.nouns(data_set['text'][i]))

# 레이블 사전 만들고 숫자 할당
label_dict = {k: i for i, k in enumerate(data_set['label'].unique())}
with open('data/label_dict.pickle', 'wb') as f:
    pickle.dump(label_dict, f)

# 데이터셋 분리
train_df, val_df = train_test_split(data_set, test_size=0.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

# 모델 설정
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(label_dict))
model.to(device)

# 옵티마이저 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 손실 함수 설정
criterion = nn.CrossEntropyLoss()

# 데이터셋 및 데이터로더 설정
train_dataset = MultiClassDataset(train_df, tokenizer, max_len=256, label_dict=label_dict)
val_dataset = MultiClassDataset(val_df, tokenizer, max_len=256, label_dict=label_dict)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)

    # 에포크 마다 손실과 정확도 출력
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {correct/total:.4f}')

# 모델 학습 후 저장
torch.save(model.state_dict(), model_save_path)