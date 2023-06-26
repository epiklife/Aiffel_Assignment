# Code Peer Review Templete
- 코더 : 김창완
- 리뷰어 : 장승우

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.

1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

- 네, 서머리와 그래프 출력이 정상적으로 되어있어요~

2.주석을 보고 작성자의 코드가 이해되었나요?

- 네, 코드 블럭마다 주석이 달려있어서 이해하기 쉬웠어요~

```python
# 훈련 데이터셋 생성
train_dataset = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'  # 이진 분류인 경우
    # class_mode='categorical'  # 다중 분류인 경우
)
```

3.코드가 에러를 유발한 가능성이 있나요?

- 없어요, 에러 조건 코드도 입력되어 있어요~
```python
    if len(num_filters) != 4:
        raise ValueError("Number of filter counts should be 4.")
```

4.코드 작성자가 코드를 제대로 이해하고 작성했나요?

- 네, 질문의 대답도 잘 해주시고 코드 조건문들이 이애할 수 있는 상태에서 짠 것으로 보여져요~
```python
   for stage, (num_blocks, filters) in enumerate(zip(layer_counts, num_filters)):
    strides = 1 if stage == 0 else 2
    x = resnet_block(x, filters, strides=strides, downsample=True, is_50=is_50)

    for _ in range(1, num_blocks):
        x = resnet_block(x, filters, is_50=is_50)
```

5.코드가 간결한가요?
- 네~ 구조를 유지하면서 진행해서 간결했어요~

# 참고 링크 및 코드 개선 여부

- validation loss 값이 초반에 높아서 그래프 표시할때 해당 부분을 제외해주시면 더 보기 좋을 것 같아요~
- 덕분에 전처리 과정에 대한 정보를 알게 되어서 좋았어요~
