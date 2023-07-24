아이펠캠퍼스 온라인4기 피어코드리뷰

Quest_19. Review date_20.July.2023

- 코더 : 김창완
- 리뷰어 : 최우정 

**PRT(PeerReviewTemplate)**


|평가문항|상세기준|완료여뷰|
|-------|--------|-------|
|1.Text recognition을 위해 특화된 데이터셋 구성이 체계적으로 진행되었다.| 텍스트 이미지 리사이징, ctc loss 측정을 위한 라벨 인코딩, 배치처리 등이 적절히 수행되었다. |각 단계마다 잘 작성하여 이미지 비율을 유지하면서도 높이만 변경하는 코드를 작성하셨습니다. 
|2. CRNN 기반의 recognition 모델의 학습이 정상적으로 진행되었다. |학습결과 loss가 안정적으로 감소하고 대부분의 문자인식 추론 결과가 정확하다. | Epoch 횟수도 평균적으로 잘 돌리셨고, CTC Loss 계산을 위한 Lambda 함수를 활용하여 결과를 잘 보여주셨습니다.
|3. keras-ocr detector와 CRNN recognizer를 엮어 원본 이미지 입력으로부터 text가 출력되는 OCR이 End-to-End로 구성되었다. |샘플 이미지를 원본으로 받아 OCR 수행 결과를 리턴하는 1개의 함수가 만들어졌다.| 원본 이미지 다음 crop 후 함수까지 잘 작성 해주셨습니다.|

** [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? ** 
네, 텍스트 이미지 리사이징 등 코드를 잘 작성 하셨습니다.

--------------------------------------------------
```python
  # 이미지는 버퍼를 통해 읽어오기 때문에 
        # 버퍼에서 이미지로 변환하는 과정이 다시 필요해요
        try:
            img = Image.open(buf).convert('RGB')

        except IOError:
            img = Image.new('RGB', (100, 32))
            label = '-'

        # 원본 이미지 크기를 출력해 봅니다
        width, height = img.size
        print('original image width:{}, height:{}'.format(width, height))
        
        # 이미지 비율을 유지하면서 높이를 32로 바꿀거에요
        # 하지만 너비를 100보다는 작게하고 싶어요
        target_width = min(int(width*32/height), 100)
        target_img_size = (target_width,32)        
        print('target_img_size:{}'.format(target_img_size))        
        img = np.array(img.resize(target_img_size)).transpose(1,0,2)

        # 이제 높이가 32로 일정한 이미지와 라벨을 함께 출력할 수 있어요       
        print('display img shape:{}'.format(img.shape))
        print('label:{}'.format(label))
        display(Image.fromarray(img.transpose(1,0,2).astype(np.uint8)))
 ```
------------------------------------------------

**[ ] 주석을 보고 작성자의 코드가 이해되었나요? **
  주석이 꼼꼼하며 추가적인 부연 설명에 대해서도 참고링크를 걸어 기재해주셨습니다.
```python
#자세히 알고 싶다면 아래 문서를 참고하세요\n",
 ```
 
** [ ] 코드가 에러를 유발할 가능성이 있나요? **
   없습니다. 
  
** [  ] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기) **
   이해를 잘 하고 계시고, 각 주석에 대해서도 어떻게 작성해야 하는지 알고 계시지만 추가 공부를 하시겠다고 하심.

** [  ] 코드가 간결한가요? **
   간결하고 저보다 더 깔끔하게 작성하신 것 같습니다. (CRNN model Build 부분의 경우 포함)


** 참고 링크 및 코드 개선 필요사항 ** 
  
  개선 하실 부분이 없습니다. 이번 퀘스트에 대해 잘 이해하고 계신 것 같습니다. 
