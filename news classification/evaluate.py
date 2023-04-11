import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from MultiClassDataset import MultiClassDataset
from konlpy.tag import Okt
okt = Okt()

# 경로 설정
test_set_path = 'data/test_set.xlsx'
trained_model_path = 'trained_models/test.pth'

# 하이퍼파라미터 설정
batch_size = 32

# 테스트 데이터셋 불러오기
test_set = pd.read_excel(test_set_path)

# 본문 명사 추출
for i in range(len(test_set['text'])):
    test_set.loc[i, 'text'] = " ".join(okt.nouns(test_set['text'][i]))

# 레이블 사전 불러오기
with open('data/label_dict.pickle', 'rb') as f:
    label_dict = pickle.load(f)

# 모델 설정
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(label_dict))
model.load_state_dict(torch.load(trained_model_path))
model.to(device)

# 데이터셋 및 데이터로더 설정
test_set.reset_index(drop=True, inplace=True)
test_dataset = MultiClassDataset(test_set, tokenizer, max_len=256, label_dict=label_dict)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 평가 루프
model.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Acc: {correct/total:.3f}')