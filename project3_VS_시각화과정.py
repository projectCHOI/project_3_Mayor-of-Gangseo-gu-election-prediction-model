# 필요한 라이브러리 임포트
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 나눔고딕 폰트 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

# 데이터 로드
df_jin = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/predictions_jin.csv")
df_kim = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/predictions_kim.csv")
df_kwon = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/predictions_kwon.csv")

# 나눔고딕 폰트 경로 설정 및 폰트 설정
path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_name = fm.FontProperties(fname=path, size=10).get_name()
plt.rc('font', family=font_name)

# 후보별 감정 분석 결과의 빈도수 계산
jin_sentiment_counts = df_jin["예측 감정"].value_counts()
kim_sentiment_counts = df_kim["예측 감정"].value_counts()
kwon_sentiment_counts = df_kwon["예측 감정"].value_counts()

# 파이 차트를 이용한 감정 분석 결과 시각화
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
labels = ["긍정", "부정"]
jin_sizes = [jin_sentiment_counts.get("긍정", 0), jin_sentiment_counts.get("부정", 0)]
kim_sizes = [kim_sentiment_counts.get("긍정", 0), kim_sentiment_counts.get("부정", 0)]
kwon_sizes = [kwon_sentiment_counts.get("긍정", 0), kwon_sentiment_counts.get("부정", 0)]

axs[0].pie(jin_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axs[0].set_title('진교훈')
axs[1].pie(kim_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axs[1].set_title('김태우')
axs[2].pie(kwon_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
axs[2].set_title('권수정')

plt.show()

# 댓글 길이 계산 및 산점도 시각화
df_kim['댓글 길이'] = df_kim['댓글'].apply(lambda x: len(str(x)))
df_jin['댓글 길이'] = df_jin['댓글'].apply(lambda x: len(str(x)))
df_kwon['댓글 길이'] = df_kwon['댓글'].apply(lambda x: len(str(x)))

plt.figure(figsize=(6, 4))
plt.scatter(df_jin['댓글 길이'], df_jin['예측 감정'], alpha=0.5)
plt.xlabel('댓글 길이')
plt.ylabel('예측 감정 (1: 긍정, 0: 부정)')
plt.title('댓글 길이에 따른 예측 감정')
plt.show()

# 긍정 및 부정 비율 계산 및 산점도/막대 그래프 시각화
# (중복되는 부분을 함수로 대체할 수 있습니다)
def plot_sentiment_ratios(df, name):
    positive_ratios = df[df['예측 감정'] == "긍정"]['댓글 길이'].value_counts() / df['댓글 길이'].value_counts()
    negative_ratios = df[df['예측 감정'] == "부정"]['댓글 길이'].value_counts() / df['댓글 길이'].value_counts()

    plt.figure(figsize=(6, 4))
    plt.scatter(positive_ratios.index, positive_ratios, label='긍정', alpha=0.5)
    plt.scatter(negative_ratios.index, negative_ratios, label='부정', alpha=0.5)
    plt.xlabel('댓글 길이')
    plt.ylabel('긍정 및 부정 비율')
    plt.legend()
    plt.title(f'{name}의 댓글 길이에 따른 긍정과 부정 비율')
    plt.show()

# 각 후보별 긍정 및 부정 비율 시각화
plot_sentiment_ratios(df_jin, '진교훈')
plot_sentiment_ratios(df_kim, '김태우')
plot_sentiment_ratios(df_kwon, '권수정')

# 후보별 댓글 수 및 중복 댓글 비율 시각화
# (데이터 준비 및 막대 그래프 생성)
x_labels = ["진교훈", "김태우", "권수정"]
y_values = [918, 986, 712]

bars = plt.bar(x_labels, y_values, color=['skyblue', 'lightcoral', 'yellow'])
for bar, y_value in zip(bars, y_values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 10, str(y_value), ha='center')

plt.title('후보별 댓글 수')
plt.ylabel('댓글 수')
plt.show()

# 후보별 댓글수 비율 파이 차트 시각화
plt.figure(figsize=(6, 6))
plt.pie(y_values, labels=x_labels, autopct='%1.1f%%', startangle=90, colors=['skyblue','lightcoral','yellow'])
plt.title('후보별 댓글수 비율')
plt.show()

# 후보별 중복 댓글 비율 막대 그래프 시각화
# (데이터 준비 및 막대 그래프 생성)
y_values = [((23061-840)/23061)*100, ((25208-970)/25208)*100, ((16857-592)/16857)*100]
bars = plt.bar(x_labels, y_values, color=['skyblue', 'lightcoral', 'yellow'])
for bar, y_value in zip(bars, y_values):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() / 2, f"{y_value:.2f}%", ha='center', va='center')

plt.title('후보별 중복 댓글 비율')
plt.ylabel('백분율')
plt.show()

# pandas 출력 옵션 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 댓글 길이에 따른 데이터 정렬 및 출력
df_jin_sorted = df_jin.sort_values(by='댓글 길이', ascending=False)
df_kim_sorted = df_kim.sort_values(by='댓글 길이', ascending=False)
df_kwon_sorted = df_kwon.sort_values(by='댓글 길이', ascending=False)

# 필터링된 댓글 출력
filtered_df_kim = df_kim[df_kim['댓글'].str.contains('김태우')]
print(filtered_df_kim["동영상 제목"].count())
filtered_df_kim[["댓글", "예측 감정"]].head(50)