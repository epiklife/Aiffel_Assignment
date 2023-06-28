# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김창완
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
문제 정의 그리고 그에대한 해결방안을 완벽하게 제시해 주셨네요! 배우고갑니다!!

### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
# 데이터 증강 augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image) # 랜덤 좌우 플립
    image = tf.image.random_brightness(image, max_delta=0.2) # [-0.2, 0.2] 만큼 이미지 밝기(픽셀값) 조정
    image = tf.image.rot90(image, k=1) # 90도 로테이션
    image = tf.image.adjust_saturation(image, 0.5) # 채도 조정
    image = tf.clip_by_value(image, 0, 1) # 0~1를 넘어가는 이미지들을 0과 1로 조정
    return image, label
```
<br> 재 사용하는 함수를 function by function으로 주석을 달아주셔서 이해가 빨랐습니다!!

### 코드가 에러를 유발할 가능성이 있나요? (X)
![image](https://github.com/epiklife/Aiffel_Assignment/assets/65104209/95f097c9-f332-4f1e-b720-421532f2ba8d)
<br> 각각 알맞게 변수를 넣고 실행함으로써 에러를 유발할 가능성은 없어보여요!
### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O) **ꉂꉂ(ᵔᗜᵔ*)**
![image](https://github.com/epiklife/Aiffel_Assignment/assets/65104209/2c237d15-419d-40a9-af12-e83531bd3bd6)
<br> 각각의 시각화를 통해 문제가 뭔지 어느 부분이 더 좋은지를 다 알고 계십니다!!

### 코드가 간결한가요? (O)
```python
# 데이터 가공 메인함수
def normalize_and_resize_img(image, label):
    # Normalizes images: `uint8` -> `float32`
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label

# 데이터 증강 augmentation
def augment(image, label):
    image = tf.image.random_flip_left_right(image) # 랜덤 좌우 플립
    image = tf.image.random_brightness(image, max_delta=0.2) # [-0.2, 0.2] 만큼 이미지 밝기(픽셀값) 조정
    image = tf.image.rot90(image, k=1) # 90도 로테이션
    image = tf.image.adjust_saturation(image, 0.5) # 채도 조정
    image = tf.clip_by_value(image, 0, 1) # 0~1를 넘어가는 이미지들을 0과 1로 조정
    return image, label

def onehot(image, label):
    label = tf.one_hot(label, num_classes)
    return image, label

def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False, with_cutmix=False, with_mixup=False):
    ds = ds.map(
        normalize_and_resize_img, 
        num_parallel_calls=2 # 병렬 데이터셋으로
    )

    if not is_test and with_aug:
        ds = ds.map(
            augment
        )
    ds = ds.batch(batch_size)
    
    if not is_test and with_cutmix:
        ds = ds.map(
            cutmix,
            num_parallel_calls=2
        )
    elif not is_test and with_mixup:
        ds = ds.map(
            mixup,
            num_parallel_calls=2
        )
    else:
        ds = ds.map(
            onehot,
            num_parallel_calls=2
        )
        
    
    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
        
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
```
<br> 수많은 함수처리를 통해 매우 간결합니다 GOOD!

----------------------------------------------

## 참고 링크 및 코드 개선
