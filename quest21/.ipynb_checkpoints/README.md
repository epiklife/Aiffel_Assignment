아이펠캠퍼스 온라인4기 피어코드리뷰 [2023-07-26]

- 코더 : 김창완
- 리뷰어 : 최우정

----------------------------------------------


## [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   
1. tfrecord를 활용한 데이터셋 구성과 전처리를 통해 프로젝트 베이스라인 구성을 확인하였습니다. 

 ```python
  
  
   def create_dataset(tfrecords, batch_size, num_heatmap, is_train):
    preprocess = Preprocessor(
        IMAGE_SHAPE, (HEATMAP_SIZE[0], HEATMAP_SIZE[1], num_heatmap), is_train)

    dataset = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(dataset)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  
  ```
2. simplebaseline 모델을 정상적으로 구현하였다.

```python

*** SIGSEGV received at time=1690339899 on cpu 3 ***
PC: @     0x7f95c87b0855  (unknown)  nsync::nsync_dll_splice_after_()
    @     0x7f962d2e53c0       1488  (unknown)
    @     0x7f95c87b087b         32  nsync::nsync_dll_make_first_in_list_()
    @     0x7f95c87b08b9         32  nsync::nsync_dll_make_last_in_list_()
    @     0x7f95c87b0ad5         96  nsync::nsync_mu_lock_slow_()
    @     0x7f95c87b0b6d         32  nsync::nsync_mu_lock()
    @     0x7f95bc4a5114         80  TF_NewOperation
    @     0x7f95b5acb198        192  pybind11::cpp_function::initialize<>()::{lambda()#3}::_FUN()
    @     0x7f95b5ac1cf0        544  pybind11::cpp_function::dispatcher()

```

->  simplebaseline 모델에서 학습을 더 시도해보고자 하셔서  문제점에 대한 해결을 그 이후에 해결이 될듯하다. 

3. Hourglass 모델과 simplebaseline 모델을 비교분석한 결과를 체계적으로 정리하였다. 

  -> 결과는 두 모델 다 나왔지만, 전반적으로 만족스러운 결과는 안나오셨다고 했다.

## [O] 주석을 보고 작성자의 코드가 이해되었나요?

  주석이 자세하게 적혀있어서 각 단계별로 이해하기 너무 편했습니다.

## [X] 코드가 에러를 유발할 가능성이 있나요?

코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.

## [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

코드보다는 구조에 더 집중하시면서, 구조에 대해 설명해주셨습니다. 

## [O] 코드가 간결한가요?
  
깔끔하게 되어있어 가독성이 좋았습니다.

----------------------------------------------

참고 링크 및 코드 개선 -> 에러 난 부분에 대해 재학습을 하면 코드 개선을 하면 되실듯합니다. 
