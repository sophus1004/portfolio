import torch

# 데이터셋 클래스 정의
class MultiClassDataset():
    def __init__(self, dataframe, tokenizer, max_len, label_dict):
        self.tokenizer = tokenizer
        self.texts = dataframe.text.values
        self.labels = [label_dict[code] for code in dataframe.label.values] # 레이블 숫자로 변환
        self.max_len = max_len
        self.label_dict = label_dict

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
