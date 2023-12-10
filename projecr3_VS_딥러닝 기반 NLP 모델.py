# 딥러닝 기반 NLP 모델을 사용하여 한국어 텍스트의 감정 분석을 수행하는 전체적인 과정
# 데이터 전처리, 모델 훈련 및 평가, 모델 저장까지의 단계를 포함

# Pytorch + HuggingFace
# KoElectra Model
# 박장원님의 KoElectra-small 사용
# https://monologg.kr/2020/05/02/koelectra-part1/
# https://github.com/monologg/KoELECTRA

# Dataset
# 네이버 영화 리뷰 데이터셋
# https://github.com/e9t/nsmc

# 필요한 라이브러리 임포트 설치
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm

# GPU 사용 설정
device = torch.device("cuda")

# NSMC 데이터셋을 처리하기 위한 클래스 정의
class NSMCDataset(Dataset):
    def __init__(self, csv_file):
        # 데이터셋 로드 및 전처리
        self.dataset = pd.read_csv(csv_file, sep='\t').dropna(axis=0)
        self.dataset.drop_duplicates(subset=['document'], inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")
        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values
        text, y = row
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y

# 훈련 및 테스트 데이터셋 로드
train_dataset = NSMCDataset("ratings_train.txt")
test_dataset = NSMCDataset("ratings_test.txt")

# 모델 정의 및 초기화
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)

# 모델 상태 로드
model.load_state_dict(torch.load("model.pt"))

# 모델 학습 설정
epochs = 5
batch_size = 16
optimizer = AdamW(model.parameters(), lr=5e-6)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# 모델 학습 과정
losses = []
accuracies = []

for i in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    model.train()

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader):
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)
        batches += 1

        if batches % 100 == 0:
            print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)

# 모델 평가
model.eval()
test_correct = 0
test_total = 0

for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
    y_batch = y_batch.to(device)
    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
    _, predicted = torch.max(y_pred, 1)
    test_correct += (predicted == y_batch).sum()
    test_total += len(y_batch)

print("Accuracy:", test_correct.float() / test_total)

# 모델 저장
torch.save(model.state_dict(), "model.pt")
