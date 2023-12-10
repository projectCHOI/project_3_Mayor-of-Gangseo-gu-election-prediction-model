# ElectraForSequenceClassification 모델을 로드
# 사전에 저장된 가중치(model.pt)를 모델에 적용

# 필요한 라이브러리 임포트
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, ElectraForSequenceClassification

# GPU 사용 설정
device = torch.device("cuda")

# 모델 불러오기 및 초기화
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator").to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# 데이터프레임 생성
df_jin = pd.read_csv("results_진교훈.csv")
df_kim = pd.read_csv("results_김태우.csv")
df_kwon = pd.read_csv("results_권수정.csv")

# 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

# 댓글 예측 함수 정의
def predict_comments(df):
    inputs = tokenizer(df["댓글"].tolist(), return_tensors='pt', truncation=True, max_length=256, pad_to_max_length=True, add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = F.softmax(outputs.logits, dim=1)
    _, predicted_labels = torch.max(predictions, 1)
    return predicted_labels.cpu().tolist()

# 각 데이터프레임의 댓글 예측
predictions_jin = predict_comments(df_jin)
predictions_kim = predict_comments(df_kim)
predictions_kwon = predict_comments(df_kwon)

# 예측 결과 출력
print("Jin 댓글 예측 결과:", predictions_jin)
print("Kim 댓글 예측 결과:", predictions_kim)
print("Kwon 댓글 예측 결과:", predictions_kwon)

# 예측 결과를 "긍정" 또는 "부정"으로 변환하는 함수 정의
def convert_to_sentiment(predictions):
    sentiment_labels = ["부정", "긍정"]
    return [sentiment_labels[prediction] for prediction in predictions]

# 예측 결과를 "긍정" 또는 "부정"으로 변환
sentiments_jin = convert_to_sentiment(predictions_jin)
sentiments_kim = convert_to_sentiment(predictions_kim)
sentiments_kwon = convert_to_sentiment(predictions_kwon)

# 예측 결과를 데이터프레임에 추가
df_jin["예측 감정"] = sentiments_jin
df_kim["예측 감정"] = sentiments_kim
df_kwon["예측 감정"] = sentiments_kwon

# 결과를 CSV 파일로 저장
df_jin.to_csv('predictions_jin.csv', index=False, encoding='utf-8')
df_kim.to_csv('predictions_kim.csv', index=False, encoding='utf-8')
df_kwon.to_csv('predictions_kwon.csv', index=False, encoding='utf-8')
