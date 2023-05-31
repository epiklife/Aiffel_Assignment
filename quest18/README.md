아이펠캠퍼스 온라인4기 피어코드리뷰 []

- 코더 : 김창완
- 리뷰어 : 최원석

----------------------------------------------


## [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
1. "너 뭐하는 놈이야"로 입력후 "저는 위로해드리는 로봇이에요 ."로 출력되었다.
2. SubwordTextEncoder를 사용하였고, 최대, 최소, 평균, 표준편차를 계산하여 통계를 출력해 시각화하였다.

## [O] 주석을 보고 작성자의 코드가 이해되었나요?

    # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
    # 예를 들어서 "나는 학생입니다." => "나는 학생 입니다 ."와 같이
    # 학생과 마침표 사이에 거리를 만듭니다.
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
.
   
   	# Q, K, V에 각각 Dense를 적용합니다
    query = self.query_dense(query)  # Q에 Dense를 적용하여 query를 계산합니다
    key = self.key_dense(key)  # K에 Dense를 적용하여 key를 계산합니다
    value = self.value_dense(value)  # V에 Dense를 적용하여 value를 계산합니다
.

    NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
    D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
    NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
    UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
    DROPOUT = 0.1 # 드롭아웃의 비율
    EPOCHS = 80 # 반복횟수
.
  
    def decoder_inference2(sentence, _new_model):
    sentence = preprocess_sentence(sentence)

    # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
    # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
    sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

     # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
     # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
     output_sequence = tf.expand_dims(START_TOKEN, 0)

      # 디코더의 인퍼런스 단계
      for i in range(MAX_LENGTH):
       # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
       predictions = _new_model(inputs=[sentence, output_sequence], training=False)
       predictions = predictions[:, -1:, :]

        # 현재 예측한 단어의 정수
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

       # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
        if tf.equal(predicted_id, END_TOKEN[0]):
          break

       # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
        # 이 output_sequence는 다시 디코더의 입력이 됩니다.
       output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

      return tf.squeeze(output_sequence, axis=0)

주석이 자세하게 적혀있어서 각 단계별로 이해하기 너무 편했습니다.


## [X] 코드가 에러를 유발할 가능성이 있나요?

코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.


## [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

코드보다는 구조에 더 집중하시면서, 트랜스포머의 구조에 대해 설명해주셨습니다. 

## [O] 코드가 간결한가요?
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

      def __init__(self, d_model, warmup_steps=2500):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

      def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

깔끔하게 되어있어 가독성이 좋았습니다.

----------------------------------------------

참고 링크 및 코드 개선
