import pandas as pd
import json

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def Train_DataBuilder(train_data_path, tokenizer):
   
    # 학습 데이터 로드
    df = pd.read_excel(train_data_path)

    # 레이블 사전 만들고 숫자 할당
    label_list = df['labels'].unique()
    label_dict = {k: i for i, k in enumerate(label_list)}
    with open('./data/label_dict.json', 'w', encoding='utf-8') as f :
        json.dump(label_dict, f, indent="\t", ensure_ascii=False)

    df['labels'] = [label_dict[code] for code in df.labels.values]

    # 데이터 분할
    train_df, val_df = train_test_split(df, train_size=0.01,test_size=0.01, random_state=42, shuffle=True, stratify=df['labels'])
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # 데이터셋 작성
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df)
    })

    # 데이터 인코딩
    encoded_train_set = dataset['train'].map(lambda x: tokenizer.encode_plus(x['text'], add_special_tokens=True, padding=True, truncation=True, max_length=512))
    encoded_val_set = dataset['validation'].map(lambda x: tokenizer.encode_plus(x['text'], add_special_tokens=True, padding=True, truncation=True, max_length=512))

    return encoded_train_set, encoded_val_set, label_dict

def Evalulate_DataBuilder(evalulate_data_path, tokenizer):
  
    # 학습 데이터 로드
    df = pd.read_excel(evalulate_data_path)

    # 레이블 사전 만들고 숫자 할당
    with open('./data/label_dict.json') as f:
        label_dict = json.load(f)

    df['labels'] = [label_dict[code] for code in df.labels.values]

    # 데이터셋 작성
    dataset = DatasetDict({
        'evaluate': Dataset.from_pandas(df)
    })

    # 데이터 인코딩
    encoded_eval_set = dataset['evaluate'].map(lambda x: tokenizer.encode_plus(x['text'], add_special_tokens=True, padding=True, truncation=True, max_length=512))

    return encoded_eval_set, label_dict