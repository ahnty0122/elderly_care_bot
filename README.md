## 맞춤형 노인 케어 반려봇 봄돌이 제작
### 1. INTRODUCTION
<img width="715" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123722576-954ce900-d8c3-11eb-9af1-b7cc9804048a.png">

### 2. DESIGN OBJECTIVES
<img width="700" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123722790-01c7e800-d8c4-11eb-9ff4-bd5491e18ad1.png">

### 3. DESIGN SOLUTIONS
#### __3-1) Dataset__
<img width="746" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123722926-43589300-d8c4-11eb-931a-33f38ff1ea48.png">

- [ETRI 인공지능 공유 플랫폼](https://nanum.etri.re.kr/share/dhkim008/robot_environment1?lang=ko_KR)

#### __3-2) Software - 포즈 예측 모델 선정__
<img width="747" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123723204-df829a00-d8c4-11eb-865f-ee54eb613fcd.png">


<img width="698" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123723432-56b82e00-d8c5-11eb-811c-0b18caab13a5.png">

<img width="748" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123723477-70f20c00-d8c5-11eb-83c6-846b532cf9d2.png">


#### __3-3) Hardware__
<img width="700" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123723838-376dd080-d8c6-11eb-8b51-146c0bf353ff.png">

- 카메라 각도 조절과 사족보행을 통한 안정적인 움직임이 가능하게 하기 위해서 각 관절마다 별도의 모터(총 13 개)를 조립
- 눈 쪽의 카메라로 이미지와 영상 수집이 가능
- 등쪽에는 버저와 LED로 원하는 값을 출력
- 등전체에는 라즈베리 파이가, 로봇의 배 부분에는 배터리가 위치
- [Freenove Robot](https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi)

#### __3-4) Raspberry Pi Server, Client Connect__
<img width="700" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123725122-80268900-d8c8-11eb-89f3-7debf0b5146e.png">


### 4. PROTOTYPES
#### 4-1) 로봇 관절 움직임 구현
- 앞, 뒤로 움직임(사족보행), 고개 끄덕임 구현

<img width="609" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123724304-1194fb80-d8c7-11eb-9a3d-47d48d57508d.png">

#### 4-2) 사용자 action에 따른 robot의 response 구현
<img width="697" alt="제목 없음" src="https://user-images.githubusercontent.com/61795757/123724904-1b6b2e80-d8c8-11eb-8817-eb6d0d2f64e8.png">
