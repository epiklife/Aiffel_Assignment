# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 김창완
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
전부 정상적으로 진행됩니다.<br>
```python
#업샘플러(디코더) 정의
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
  
  '''
  
  !tensorboard dev upload --logdir {log_dir}
```
<br>- 인코더 디코더를 깔끔한 함수로 구현해 좀더 직관적인 코드가 됐습니다
<br>- 로그를 파일로 저장하는 점이 관리하기 더 편한거같아요!<br>
![image](https://github.com/epiklife/Aiffel_Assignment/assets/65104209/929b9858-4152-496b-b549-d0851dde9b22)
<br>모델 아키텍쳐를 시각화를 해서 Encoder Decoder방식을 조금더 간편하게 이해했습니다!

### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
```
<br>각 함수 옆에 결과로 나오는 배치사이즈를 나열함으로 써 실행되는 플로우방식을 이해하기 훨씬 편한거같아요! 

### 코드가 에러를 유발할 가능성이 있나요? (X)
코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.
![image](https://github.com/epiklife/Aiffel_Assignment/assets/65104209/c6980dd9-f266-4f0c-b83e-03646787160e)

<br>해당 코드에는 실행오류가 났지만 노드 상 학습이 너무 오래걸려 중단한 오류라 실행과는 전혀 상관없습니다!

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
![image](https://github.com/epiklife/Aiffel_Assignment/assets/65104209/f30facc2-af2d-496f-ab7f-50273684a498)
<br>구조를 시각화하면서 완벽히 파악하고 계십니다!
### 코드가 간결한가요? (O)
```python
down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
```
단순하게 표현한것이 결국 간결하기 떄문에 위 코드는 훨씬 간결한 코드인거같아요!<br>

----------------------------------------------

## 참고 링크 및 코드 개선
