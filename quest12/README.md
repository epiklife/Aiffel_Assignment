## Code Peer Review Template
---
* 코더 : 김창환
* 리뷰어 : 정연준 


PRT(PeerReviewTemplate)
---
- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

프로젝트의 목표  
[o] 1. 프로젝트1 : MSE 값이 3000 이하
훈련 세트
Iteration 40000 : Loss 2874.4185
테스트 세트 MSE
2890.7591130620294

![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/3e1c2373-d6bc-4c34-a9f3-0da711e32b69)


[o] 1. 프로젝트2 : RMSE 값이  150 이하 
테스트 세트 RMSE 결과
140.1503291812959

![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/578d6082-8168-4a2e-8fb2-b3bc296f41e0)
  
[o] 1. 시각화 여부  
![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/a748d1fd-d764-46a3-9b86-e14b916a8a69)
![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/906ab089-dbe1-4f8f-ad2a-c463654b0292)
![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/94ba3995-9710-482a-85e6-190b5ceab42c)
![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/4e1b2eac-8831-4283-8375-900e9783577f)



- [x] 주석을 보고 작성자의 코드가 이해되었나요?
 * 주석내용 미기재  
- [x] 코드가 에러를 유발할 가능성이 있나요?
 * 에러유발 가능성 없음 (직관적인 흐름에 따른 코드 작성)
- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
 * 소수점자리수 표기법에 대한 이해 
 
 * 열 슬라이싱방법 숙지
X = train[['year', 'month', 'day', 'hour', 'minute', 'second', 'temp', 'atemp', 'humidity']]
y = train['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)



- [o] 코드가 간결한가요?
 * 전체적으로 코드 반복 없어 간결함






