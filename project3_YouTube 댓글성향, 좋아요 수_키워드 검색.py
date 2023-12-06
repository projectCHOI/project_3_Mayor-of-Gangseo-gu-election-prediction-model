# 유튜브 크롤링
#Youtube API Key를 받아야 함.
#Youtube API Key : ##Key##

#pip 유튜브 키를 사용해 크롤링 모듈을 가져온다.

import os #운영 체제 관련 작업을 시작한다.

#google의 API 사용을 위한 코드, 
import google_auth_oauthlib.flow # API OAuth 인증
import googleapiclient.discovery # API 서비스 호출
import googleapiclient.errors    # API 오류 처리

import pandas as pd #데이터 처리

import time #하나의 작업이 끝나면 10초의 딜레이를 건다. 아래에 time.sleep(시간 초)추가

#작업 시작~유튜브를 불러 오는 것.
# YouTube API 키를 설정
api_service_name = "youtube"
api_version = "v3"
api_key = "##Key##"  # 여기에 API 키를 넣자.

# YouTube API 클라이언트를 생성
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=api_key)

#크롤링1 하나의 검색어
search_query = "찾고자 하는 단어"

#크롤링2 두개 이상의 검색어 조합
#candis = ['바나나', '사과', '샤인머스켓']

#for candi in candis:
#    search_query = f"내가 구운 {candi}"
#이렇게 하면 이렇게 뜬다.
#내가 구운 {바나나}
#내가 구운 {사과}
#내가 구운 {샤인머스켓}

    # YouTube 동영상 검색을 위한 요청을 만듭니다
search_request = youtube.search().list(
        q=search_query,
        type="video", #type을 비디오만
        part="id", #part는 비디오 안에 id만
        maxResults=300 #maxResults 이건 동영상의 갯수
    )

    # API 요청을 실행하고 응답을 받습니다
search_response = search_request.execute()

    #여기서 부터는 for문 사용
    # 검색 결과를 데이터프레임으로 변환합니다
video_data = []
for item in search_response["items"]:
    video_id = item["id"]["videoId"]

    #동영상 정보 요청
    #https://www.youtube.com/watch?v=9zc57ArYET4
    #https://www.youtube.com/watch?v=는 비디오 [뒤에 코드] 비디오 아이디
    video_request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )

    # API 요청을 실행하고 응답을 받습니다
    try:
        comments_response = comments_request.execute()
    except googleapiclient.errors.HttpError as e:
        error_content = e.content.decode("utf-8")
        if e.resp.status == 403 and 'commentsDisabled' in error_content:
            # 해당 동영상은 댓글이 비활성화되어 있음
            print(f"댓글이 비활성화된 동영상: {video_id}")
            continue  # 다음 동영상 처리로 넘어갑니다
        else:
            # 다른 오류 처리
            raise e


    # 가져온 댓글 정보를 리스트에 저장합니다
    for comment_item in comments_response["items"]:
        comment_text = comment_item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comment_author = comment_item["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]
        comment_publish_date = comment_item["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
        comment_likes = comment_item["snippet"]["topLevelComment"]["snippet"]["likeCount"]

        video_data.append([video_title, video_publish_date, video_likes, comment_text, comment_author, comment_publish_date, comment_likes])

    # 데이터프레임을 생성합니다
    columns = ["동영상 제목", "게시일", "영상 좋아요 수", "댓글", "작성자", "댓글 작성일", "댓글 좋아요 수"]
    merged_df = pd.DataFrame(video_data, columns=columns)

    # 데이터프레임을 CSV 파일로 저장합니다
    merged_df.to_csv(f"동작구_{candi}.csv", index=False, encoding="utf-8-sig")

    print("검색 결과를 CSV 파일로 저장했습니다.")
    
    #time.sleep 하나의 작업이 끝나면 10초의 딜레이를 건다. 위에 time 인포트 해야함
    time.sleep(10)

# 데이터프레임 출력
df

for candi in candis:
# Excel 파일을 판다스 데이터프레임으로 읽어오기 ('utf-8' 인코딩을 사용한 경우)
 df = pd.read_csv(f"동작구_{candi}.csv")