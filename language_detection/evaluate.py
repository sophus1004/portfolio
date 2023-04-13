import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from MultiClassDataset import MultiClassDataset

# 경로 설정
test_set_path = ''
save_result_path = ''
trained_model = 'trained_models'

# 하이퍼파라미터 설정
batch_size = 32

# 테스트 데이터셋 불러오기 & 행 램덤으로 재배열
test_set = pd.read_excel(test_set_path)
test_set = test_set.sample(frac=1, random_state=42)

# 레이블 사전 불러오기
with open('data/label_dict.pickle', 'rb') as f:
    label_dict = pickle.load(f)

# 모델 설정
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForSequenceClassification.from_pretrained(trained_model, num_labels=len(label_dict))
model.to(device)

# 데이터셋 및 데이터로더 설정
test_set.reset_index(drop=True, inplace=True)
test_dataset = MultiClassDataset(test_set, tokenizer, max_len=128, label_dict=label_dict)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 평가 루프
model.eval()
val_loss = 0
correct = 0
total = 0
label_list = list(label_dict.keys())
pred_list = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for pred in predicted:
            pred_list.append(label_list[pred])

print(f'Test Acc: {correct/total:.3f}')

# 에측 결과를 엑셀 파일로 저장
test_set['prediction'] = pred_list
test_set.to_excel(save_result_path)