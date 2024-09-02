# project_3_Mayor-of-Gangseo-gu-election-prediction-model

## 프로젝트 개요
- 이 프로젝트는 유튜브 댓글에 대한 감정 분석을 통해 2023년 강서구청장 보궐선거의 결과를 예측하는 것을 목표로 한다.
- 미국 44대 대통령 버락 오바마의 선거 전략이 빅데이터에 기반하여 수립된 것처럼, 이번 프로젝트도 빅데이터 분석을 통해 선거 결과를 예측한다.
- 국내 대다수가 사용하는 유튜브 데이터를 활용하여 더불어민주당 진교훈 후보의 당선을 예측했으며, 실제 선거 결과에서도 진교훈 후보가 당선되었다.

## 프로젝트 목표
후보자에 대한 대중의 감정을 분석하여, 선거 결과에 대한 통찰을 제공
- 데이터 수집 : 강서구청장 선거 후보자와 관련된 유튜브 댓글 데이터를 수집
- 감정 분석 : 수집된 댓글 데이터를 바탕으로 딥러닝 기반의 자연어 처리(NLP) 모델을 사용해 감정 분석을 수행
- 결과 예측 : 댓글의 감정 분석 결과를 토대로 선거 결과를 예측

## 사용 기술
- 프로그래밍 언어 : Python (VSCode, Google Colab)
- 라이브러리 및 도구 :
  - googleapiclient : YouTube API와의 상호작용
  - pandas : 데이터 처리 및 분석
  - matplotlib : 데이터 시각화
  - time : API 요청 시의 딜레이를 관리

## 데이터 수집
- 수집 기간 : 2023년 5월 ~ 2023년 10월 (5개월)
- 수집 대상 후보자 : 출마 후보자 "권수정", "김태우", "진교훈"
- 수집 데이터 항목 : 동영상 제목 / 게시일 / 영상 좋아요 수 / 댓글 / 작성자 / 댓글 작성일 / 댓글 좋아요 수

## 프로젝트 진행 흐름
- 데이터 수집 > 전처리 > 데이터 표준화 > 자연어 처리(NLP) > Sentiment analysis > 시각화

1. 데이터 수집
   - YouTube API를 사용하여 후보자와 관련된 영상과 댓글 데이터를 수집
   - 수집된 데이터는 CSV 형식으로 저장
2. 데이터 전처리
   - 수집된 댓글 데이터를 정제하고 전처리
   - 일관된 분석을 위해 텍스트 데이터를 표준화
3. 자연어 처리(NLP)
   - NLP 기술을 적용하여 댓글의 감정을 분석
   - KoELECTRA 모델을 사용하여 감정 분석을 수행
4. 감정 분석
   - KoELECTRA 모델을 활용하여 댓글을 긍정 또는 부정으로 분류
   - 각 후보자에 대한 전반적인 감정을 분석
5. 시각화
   - `pyLDAvis`와 Google Looker Studio를 사용해 결과를 시각화
   - PPT를 통해 분석 결과를 요약하여 프레젠테이션 제작

## 감정 분석 모델
- KoELECTRA : 이 프로젝트에서는 감정 분석을 위해 KoELECTRA 모델을 사용
- [KoELECTRA GitHub 저장소 : https://github.com/monologg/KoELECTRA]

## 시각화 도구
- pyLDAvis : 토픽 모델링 결과를 시각화
- Google Looker Studio : 대시보드 및 시각 분석
- PowerPoint : 결과를 명확하고 간결하게 프레젠테이션

## 제한점
1. 댓글 분류의 한계 : 특정 후보자의 응원 댓글이라도 다른 후보자의 영상에 달린 경우 긍정적인 댓글로 분류될 가능성
2. 변수 반영의 한계 : 선거 결과에 영향을 미치는 다양한 변수들(예: 여론조사 결과, 뉴스 기사 댓글 등)이 반영되지 않음

## 향후 개선 방향
1. 모델 개선 : 향후 KcELECTRA 모델을 사용하여 댓글 내에서 후보자의 이름까지 구분하여 긍정/부정을 정확하게 분류할 수 있도록 모델을 개선
2. 다양한 변수 반영 : 여론조사 결과, 뉴스 기사 댓글 등 선거에 영향을 미치는 다양한 변수를 반영하여 예측 모델의 정확도를 향상
- Git connection verification
