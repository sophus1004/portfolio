from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from kobert_tokenizer import KoBERTTokenizer

from CustomCallback import CustomCallback
from CustomMetrics import CustomMetrics
from DataBuilder import Train_DataBuilder, Evalulate_DataBuilder

class ModuleClassification():

    def train():
    # 학습 데이터 로드
        train_data_path = 'data/train_data.xlsx'

        # 토크나이저와 모델 로드
        # 'skt/kobert-base-v1'
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        encoded_train_set, encoded_val_set, label_dict = Train_DataBuilder(train_data_path, tokenizer)
        model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=len(label_dict))

        # 훈련 인자(TrainingArguments) 설정
        training_args = TrainingArguments(
            output_dir = './trained_models',  # 훈련 결과 출력 디렉토리
            logging_dir = './trained_models/logs',  # 로그 저장 디렉토리
            logging_steps = 1,  # 로그 출력 주기
            num_train_epochs = 1,  # 훈련할 에폭 수
            per_device_train_batch_size = 8,  # 하나의 gpu에 할당 될 배치 크기
            per_device_eval_batch_size = 8,  # 하나의 gpu에 할당 될 평가 배치 크기
            optim = 'adamw_torch',  # 옵티마이저 설정 (TrainingArguments future version에서 변경 예정)
            learning_rate = 1e-4,  # 학습률
            weight_decay = 1e-4,  # 가중치 감소
            do_eval = True,
            evaluation_strategy = 'epoch',
            save_strategy = 'epoch',
            load_best_model_at_end = True,
            metric_for_best_model = 'eval_f1',
            disable_tqdm = True,
            use_mps_device = True
        )

        # Trainer 객체 초기화
        trainer = Trainer(
            model = model,
            tokenizer = tokenizer,  # 훈련할 모델
            args = training_args,  # 훈련 인자
            train_dataset = encoded_train_set,  # 인코딩된 훈련 데이터셋
            eval_dataset = encoded_val_set,
            compute_metrics = CustomMetrics.compute_metrics,
            callbacks = [CustomCallback(), EarlyStoppingCallback(early_stopping_patience=5)] # metric_for_best_model의 로그값을 읽어 와서 선택
        )

        trainer.train()
        trainer.save_model('./best_model')

    def evalulate():
    # 학습 데이터 로드
        evalulate_model_path = './best_model'
        evalulate_data_path = 'data/train_data.xlsx'

        # 토크나이저와 모델 로드
        tokenizer = KoBERTTokenizer.from_pretrained(evalulate_model_path)
        encoded_eval_set, label_dict = Evalulate_DataBuilder(evalulate_data_path, tokenizer)
        model = BertForSequenceClassification.from_pretrained(evalulate_model_path, num_labels=len(label_dict))

        # Trainer 객체 초기화
        trainer = Trainer(
            model = model,
            tokenizer = tokenizer,  # 훈련할 모델
            eval_dataset = encoded_eval_set,
            compute_metrics = CustomMetrics.compute_metrics,
            callbacks = [CustomCallback()]
        )

        trainer.evaluate()
        