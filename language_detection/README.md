# Lv. 1 Language detection

📝 [https://www.notion.so/Language-detection-c2173486c0ea4751a41832af1a6325f4](https://www.notion.so/Lv-1-Language-detection-c2173486c0ea4751a41832af1a6325f4)

🐈‍⬛ [https://github.com/sophus1004/portfolio/tree/main/language_detection](https://github.com/sophus1004/portfolio/tree/main/language_detection)

# 개요

**다국어를 지원하는 번역 모델 등은 임의의 텍스트가 입력되면 입력된 텍스트가 어느 나라의 언어이지 자동으로 인식하는 기능이 필요하다. 이러한 자동 인식 모델 등은 이미 fasttext-langdetect, pycld3 등 배포되고 있는 라이브러리가 있지만 지원되는 언어가 제한적이고 특정한 언어에서 오답이 편중되는 경향을 보여준다. 이번 프로젝트에서는 다국어 토크나이저를 지원하는 'bert-base-multilingual-cased’ 를 사용해 언어 추론 모델을 작성한다.**

# 모델 구조

### 개발 환경 : M1 MacBook

**사용 라이브러리 리스트**

```python
## train.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from MultiClassDataset import MultiClassDataset
```

```python
## evaluate.py
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from MultiClassDataset import MultiClassDataset
```

### 하이퍼파라미터, 데이터셋 설정

```python
# 경로 설정
train_set_path = 'data/train_set.xlsx'
model_save_path = 'trained_models'

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 10
learning_rate = 1e-6

# 학습 데이터셋 불러오기
data_set = pd.read_excel(train_set_path)

# 레이블 사전 만들고 숫자 할당
label_dict = {k: i for i, k in enumerate(data_set['label'].unique())}
with open('data/label_dict.pickle', 'wb') as f:
    pickle.dump(label_dict, f)

# 데이터셋 분리
train_df, val_df = train_test_split(data_set, test_size=0.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
```

- 학습데이터는 xlsx파일을 사용해 pandas 데이터 프레임으로 만든다.
- 모델 저장은 ‘save_pretrained()’ 메서드를 사용하므로 경로는 폴더로 지정해 준다.
- 학습 데이터의 ‘label’에서 자동으로 레이 종류를 추출해 숫자를 할당하고 딕셔너리로 만든다.
- ‘evaluate.py’에서 같은 레이블 사전을 사용하기 위해 딕셔너리를 파일로 저장한다.
- 데이터 프레임을 학습용(train)과 검증용(validation)으로 나누고 순서를 섞어 준다.

### 모델과 학습 세부사항 설정

```python
# 모델 설정
device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=len(label_dict))
model.to(device)

# 옵티마이저 설정
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 손실 함수 설정
criterion = nn.CrossEntropyLoss()
```

- 개발 환경은 M1 맥북이므로 M1용 GPU 설정을 한다.
- 토크나이저는 102개 국어를 지원하는 'bert-base-multilingual-uncased'를 사용한다.
- 모델은 ‘transformers’ 라이브러리에서 지원하는 ‘BertForSequenceClassification’ 사전학습 모델을 사용한다. ‘BertForSequenceClassification’은 'bert-base-multilingual-uncased'를 지원하지 않아 지원하지 않는 가중치에 대한 에러가 발생하지만 파인 튜닝 학습에서는 문제가 되지 않는다.
- 옵티마이저는 AdamW를 사용한다.
- 손실 함수는 분류모델에 주로 사용되는 ‘CrossEntropy’를 사용한다.

### 데이터셋 학습 입력 형식으로 변환

```python
# 데이터셋 및 데이터로더 설정
train_dataset = MultiClassDataset(train_df, tokenizer, max_len=128, label_dict=label_dict)
val_dataset = MultiClassDataset(val_df, tokenizer, max_len=128, label_dict=label_dict)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

- ‘MultiClassDataset’에서는 데이터프레임의 텍스트 데이터를 토크나이징 하고 레이블 데이터를 앞서 할당된 숫자로 변환한다. 반환값은 'input_ids: 토큰 ID 시퀀스', 'attention_mask: 어텐션 마스크 시퀀스', 'labels: 레이블 시퀀스' 이다.
- ‘DataLoader’에서는 ‘MultiClassDataset’에서 반환된 값들을 학습 루프에 입력하기 위해 설정된 배치 사이즈로 나누어 반환한다.

### 학습

```python
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
```

- ‘model.train()’로 모델을 학습 모드로 전환한다.
- ‘optimizer.zero_grad()’로 모델의 가중치 기울기를 초기화 해준다.
- ‘outputs = model(input_ids, attention_mask=attention_mask, labels=labels)’로 예측된 레이블과 손실함수를 반환 받는다.
- ‘loss = criterion(outputs.logits, labels)’에서 손실값을 반환 받는다.
- ‘loss.backward()’에서 손실값으로 역전파를 수행하고 optimizer.step()으로 가중치와 편향을 업데이트한다.
- 학습 후 모델을 model.eval()로 예측 모드로 전환하고 검증(validation) 연산을 실행한다.
- 검증 연산에서는 가중치는 업데이트 하지 않고 예측만 해서 수행해서 모델의 성능을 가한다.

### 모델 저장

```python
# 모델 학습 후 저장
torch.save(model.state_dict(), 'torch_model.pt')
model.save_pretrained(model_save_path)
```

# 결과

### 사용한 데이터셋

| label | text |
| --- | --- |
| ar | التحدث إلى شخص آخر بلغة مختلفة يمكن أن يساعد على فتح ذهنك وتطوير مهارات التواصل الخاصة بك. |
| de | Das Reisen kann helfen, die Welt und die unterschiedlichen Kulturen besser zu verstehen und zu schätzen zu lernen. |
| en | Reading books can take you to different places and times, introducing you to new ideas and perspectives. |
| es | El aprendizaje de una nueva lengua puede abrir puertas a nuevas oportunidades personales y profesionales. |
| fr | Le temps passé avec des amis et la famille est précieux et peut créer des souvenirs durables. |
| id | Menerima kesalahan sebagai bagian dari belajar dapat membantu meningkatkan kepercayaan diri dan kemampuan seseorang. |
| ja | 日本の春の桜の花は美しく、その季節を祝う文化的な行事を楽しむことができます。 |
| ko | 새로운 일을 시도하는 것은 언제나 두려움을 동반하지만, 그것이 당신의 인생을 변화시킬 수도 있다. |
| ru | Чтение книг может расширить ваш кругозор и помочь понимать иностранные культуры и людей лучше. |
| ti | Ang pagtitiwala sa sarili ay mahalaga upang magtagumpay sa buhay, kahit na may mga pagsubok na darating. |
| uz | Yaxshi dostlar va oila bilan o'tgan vaqt bahol va unga qolgan xotiralar o'ziga xos vaqt-maqsadning nishoni bo'ladi. |
| vi | Các giá trị gia đình, như tình yêu và sự chăm sóc, có thể giúp chúng ta vượt qua khó khăn trong cuộc sống. |
| zh-cn | 旅行可以让你开阔视野，了解不同的文化，结识新朋友并创造难忘的回忆。 |
- 데이터셋은 위와 같은 데이터 형식으로 13개 국어를 각각 1000 문장씩 사용했다.
- 총 13000 문장 중 10400 문장은 학습용으로 2600 문장은 테스트용으로 사용했다.

### 결과

- 검증 정확도(validation accuracy) 99.9%
- 테스트 정확도 (test accuracy) 100%

# 이슈와 해결

### 검증 정확도가 올라가지 않음

- 검증 확도가 78%이상 올라가지 않는 문제가 발생
    
    >> 레이블이 13개이다 보니 각 레이블 예측도가 큰 차이를 보여야 제대로 된 예측이 가능하다. 예를 ‘id’와 ‘vi’가 항상 비슷한 확률로 예측이 된다면 ‘Learning rate’의 값을 줄여 좀 더 깊은 최소값 지점을 찾을 필요가 있다.
    
- ‘Learning rate’를 ‘1e-4’에서 ‘1e-6’으로 낮추어 해결

### 예상보다 높은 정확도

- 적은 학습량에도 예측 정확도 100%를 가진 모델이 만들어 짐
    
    >> 사전학습 모델로 사용된 'bert-base-multilingual-uncased' 102개 국어 각각의 문장으로 MLM과 NSP 학습이 되어 있는 모델이다. 트랜스포머의 셀프 어텐션을 사용한 MLM 학습은 각 토큰의 문맥적 상관관계에 대한 학습을 임베딩 값을 가지므로 분류 모델에서 업데이트 되는 임베딩 값에 그대로 반영된 것으로 보인다.
    
    [bert-base-multilingual-uncased · Hugging Face](https://huggingface.co/bert-base-multilingual-uncased)