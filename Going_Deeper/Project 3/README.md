# Code Peer Review Templete
- 코더 : 김창완
- 리뷰어 : 장승우

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 체크하고 확인하여 작성한 코드에 적용하세요.
- [⭕] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

네, 이미지 결과가 정상적으로 나왔어요~

![image](https://github.com/epiklife/Aiffel_Assignment/assets/131636630/0ea712a1-5b46-40fb-9784-a0716ed51712)

- [⭕] 2.주석을 보고 작성자의 코드가 이해되었나요?

네, 주석을 잘 달아주셔서 이해하기 쉬웠어요~

```python
def getGrad_CAM_Iou(base_model, layer_name, item, alpha=0.5, box_th=0.25):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8)) # 2 x 2 이미지 틀 생성
    axes = axes.ravel() # 2차원 배열을 1차원으로 변환
    origin_image = item['image'].astype(np.uint8) # org_img
```

- [❌] 3.코드가 에러를 유발한 가능성이 있나요?

변수에 직접 정의해서 함수에 넣기때문에 에러를 유발할 가능성은 낮아요~

```python
layer_name = 'conv5_block3_out'
box_th = 0.25
alpha = 0.7
iou = getGrad_CAM_Iou(base_model, layer_name, item, alpha, box_th)
print('iou = ', iou)
```

- [⭕] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?

네, 질문에 대답도 잘 해주시고, 직접 함수도 작성해서 활용했어요~

```python
def getGrad_CAM_Iou(base_model, layer_name, item, alpha=0.5, box_th=0.25):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8)) # 2 x 2 이미지 틀 생성
    axes = axes.ravel() # 2차원 배열을 1차원으로 변환
    origin_image = item['image'].astype(np.uint8) # org_img
...
    
    plt.tight_layout()
    plt.show()
    
    grad_cam_pred_bbox = rect_to_minmax(grad_cam_rect, item['image'])
    
    return  get_iou(grad_cam_pred_bbox, item['objects']['bbox'][0])
```

- [⭕] 5.코드가 간결한가요?

네, 직관적으로 이해할 수 있게 작성해주셨어요~

```python
    origin_image = item['image'].astype(np.uint8) # org_img
    
    axes[0].imshow(origin_image) # img 설정
    axes[0].axis('off') # 축 제거
    axes[0].set_title('org_img') # title 설정
    
    grad_cam_image = generate_grad_cam(base_model, layer_name, item) # grad_cam 이미지 
    
    axes[1].imshow(grad_cam_image) # img 설정
    axes[1].axis('off') # 축 제거
    axes[1].set_title('grad_cam_img') # title 설정
```

# 참고 링크 및 코드 개선 여부
같은 이미지로 cam,grad_cam을 비교했으면 더 좋았을 것 같아요~

cam에서 특이하게 나온 이미지를 첨부해주셔서 알아갈 내용이 생겨서 좋았어요~
