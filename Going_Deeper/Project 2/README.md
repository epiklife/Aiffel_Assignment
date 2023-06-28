# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김신성
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
각각의 모델들이 정상적으로 작동하였고 비교문제를 정확히 해결하였습니다.
```python
aug_resnet50.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'],
)
```
<br>
loss함수또한 categorical_crossentropy로 변경한 부분도 멋집니다!
<br>
![image](https://github.com/epiklife/Aiffel_Assignment/assets/65104209/228fc2e6-17f6-4f4d-bbd0-9804e3c445c3)
<br>
문제 정의또한 완벽하게 되었습니다.

### 주석을 보고 작성자의 코드가 이해되었나요? (O)

### 코드가 에러를 유발할 가능성이 있나요? (X)

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O) **ꉂꉂ(ᵔᗜᵔ*)**

### 코드가 간결한가요? (O)

----------------------------------------------

## 참고 링크 및 코드 개선
