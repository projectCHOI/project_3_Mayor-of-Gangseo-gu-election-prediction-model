import pandas as pd

# 후보자 목록
candis = ['진교훈', '김태우', '권수정']

# 각 후보자의 결과 데이터 CSV 파일 읽기 및 기술적 분석 출력
for candi in candis:
    # CSV 파일 읽기
    file_path = f"/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/results_{candi}.csv"
    df = pd.read_csv(file_path)

    # 기술적 통계 출력
    print(df.describe())
    print()

# 각 후보자의 엑셀 파일 읽기 및 CSV로 변환
for candi in candis:
    # 엑셀 파일 읽기
    excel_file_path = f"/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/Raw Data/comment_{candi}.xlsx"
    excel_data = pd.read_excel(excel_file_path)

    # CSV 파일로 저장
    csv_file_path = f"/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/Raw Data/comment_{candi}.csv"
    excel_data.to_csv(csv_file_path, index=False)

# 변환 완료 메시지
print('성공적으로 변환되었습니다.')

# 각 후보자별 댓글 데이터 CSV 파일 읽기
df_jin = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/Raw Data/comment_진교훈.csv", header=None)
df_kim = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/Raw Data/comment_김태우.csv", header=None)
df_kwon = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/Raw Data/comment_권수정.csv", header=None)

# 기술적 분석 출력
print(df_jin.describe())
print(df_kim.describe())
print(df_kwon.describe())

# 열 이름 변경 및 'comment_length' 열 추가
df_jin.columns = ["comment"]
df_kim.columns = ["comment"]
df_kwon.columns = ["comment"]

df_jin['comment_length'] = df_jin['comment'].apply(lambda x: len(str(x)))
df_kim['comment_length'] = df_kim['comment'].apply(lambda x: len(str(x)))
df_kwon['comment_length'] = df_kwon['comment'].apply(lambda x: len(str(x)))

# 기술적 분석 재출력
print(df_jin.describe())
print(df_kim.describe())
print(df_kwon.describe())

# Pandas 출력 옵션 설정
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 'comment_length' 열을 기준으로 정렬하여 상위 10개 행 출력
sorted_df_jin = df_jin.nlargest(80, 'comment_length', keep='all')
print(sorted_df_jin)

# 특정 값 선택
selected_value = df_jin.loc[[11782, 1736, 862], 'comment']  # 특정 인덱스의 'comment' 열 값 선택
print(selected_value)

# 중복된 행의 개수 계산 및 출력
duplicate_rows_jin = df_jin['comment'].duplicated().sum()
duplicate_rows_kim = df_kim['comment'].duplicated().sum()
duplicate_rows_kwon = df_kwon['comment'].duplicated().sum()

print(f'jin 중복된 행의 개수: {duplicate_rows_jin}')
print(f'kim 중복된 행의 개수: {duplicate_rows_kim}')
print(f'kwon 중복된 행의 개수: {duplicate_rows_kwon}')

import pandas as pd

# 후보자별 결과 데이터 CSV 파일 읽기
df_jin = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/results_진교훈.csv")
df_kim = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/results_김태우.csv")
df_kwon = pd.read_csv("/content/drive/MyDrive/SeSAC_AI 전문인력 양성/Project/2023년 강서구청장 보궐선거 예측분석/Raw Data/results_권수정.csv")

# 각 데이터프레임의 기술적 분석 출력
print("진교훈")
print(df_jin.describe())
print()

print("김태우")
print(df_kim.describe())
print()

print("권수정")
print(df_kwon.describe())
print()

# 후보자별 '댓글' 열의 기술적 분석
print("진교훈")
print(df_jin["댓글"].describe())
print()

print("김태우")
print(df_kim["댓글"].describe())
print()

print("권수정")
print(df_kwon["댓글"].describe())
print()

# 후보자별 '작성자' 열의 중복 빈도수 계산
duplicate_counts_jin = df_jin['작성자'].value_counts()
duplicate_counts_kim = df_kim['작성자'].value_counts()
duplicate_counts_kwon = df_kwon['작성자'].value_counts()

# 중복 빈도수 출력
print("진교훈")
print(duplicate_counts_jin)
print()

print("김태우")
print(duplicate_counts_kim)
print()

print("권수정")
print(duplicate_counts_kwon)
print()
