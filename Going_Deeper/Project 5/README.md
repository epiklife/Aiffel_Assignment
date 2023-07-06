# AIFFEL GoingDeeper
----  
## **Code Peer Review**
------------------
- 코더 : 김창완
- 리뷰어 : 김설아

## **PRT(PeerReviewTemplate)**  
------------------  
- [x] **1. 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
- [x] **2. 주석을 보고 작성자의 코드가 이해되었나요?**  
 ```python
def plus_build_model(input_shape=(224, 224, 3)):
    # 입력 레이어
    inputs = Input(input_shape)

    # 첫 번째 Contracting Path
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 두 번째 Contracting Path
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 세 번째 Contracting Path
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 네 번째 Contracting Path
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Expanding Path
    up4 = UpSampling2D(size=(2, 2))(pool4)
    up4 = Conv2D(256, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv4, up4], axis=3)
    conv5 = conv_block(merge4, 512)

    up3 = UpSampling2D(size=(2, 2))(conv5)
    up3 = Conv2D(128, 2, activation='relu', padding='same')(up3)
    merge3 = concatenate([conv3, up3], axis=3)
    conv6 = conv_block(merge3, 256)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = Conv2D(64, 2, activation='relu', padding='same')(up2)
    merge2 = concatenate([conv2, up2], axis=3)
    conv7 = conv_block(merge2, 128)

    up1 = UpSampling2D(size=(2, 2))(conv7)
    up1 = Conv2D(32, 2, activation='relu', padding='same')(up1)
    merge1 = concatenate([conv1, up1], axis=3)
    conv8 = conv_block(merge1, 64)

    # 출력 레이어
    output = Conv2D(1, 1, activation='sigmoid')(conv8)

    # 모델 생성
    model = Model(inputs=inputs, outputs=output)

    return model
 ```
 > 단계 별로 주석을 달아주셔서 이해가 수월했습니다.

- [x] **3. 코드가 에러를 유발할 가능성이 있나요?**
 ```python
def build_augmentation(is_train=True):
  if is_train:    # 훈련용 데이터일 경우
    return Compose([
                    HorizontalFlip(p=0.5),    # 50%의 확률로 좌우대칭
                    RandomSizedCrop(         # 50%의 확률로 RandomSizedCrop
                        min_max_height=(300, 370),
                        w2h_ratio=370/1242,
                        height=224,
                        width=224,
                        p=0.5
                        ),
                    Resize(              # 입력이미지를 224X224로 resize
                        width=224,
                        height=224
                        )
                    ])
  return Compose([      # 테스트용 데이터일 경우에는 224X224로 resize만 수행합니다. 
                Resize(
                    width=224,
                    height=224
                    )
                ])
 ```
 > 여러 함수에서 조건에 따른 분류를 통해 에러를 방지하셨습니다.

- [x] **4. 코드 작성자가 코드를 제대로 이해하고 작성했나요?**  
  
 ```python
def build_model(input_shape=(224, 224, 3)):
    # 입력 레이어
    inputs = Input(shape=input_shape)

    # Contracting Path
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Expanding Path
    up4 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3)
    merge4 = concatenate([conv2, up4])
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = concatenate([conv1, up5])
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # 출력 레이어
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5) # 이 예시에서는 이진 분류를 위해 sigmoid 사용

    # 모델 생성
    model = Model(inputs=inputs, outputs=outputs)

    return model
 ```
 > 모델을 이해하고 데이터 셋에 맞게 수정하셨습니다.

- [x] **5. 코드가 간결한가요?**  
  
 ```python
def conv_block(inputs, filters, kernel_size=3):
    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x
 ```
 > 반복되는 연산을 함수로 처리해서 간결하게 표현하셨습니다.

## **참고링크 및 코드 개선 여부**  
------------------  
- 
    
