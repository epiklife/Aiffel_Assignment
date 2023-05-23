## Code Peer Review Template
---
* 코더 : 김창완
* 리뷰어 : 정연준


## PRT(PeerReviewTemplate)
---
- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
```
sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
# img_bgr은 rgb로만 적용해놓은 원본 이미지 
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker>127,sticker_area,img_sticker).astype(np.uint8), 0.5, 0)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) 
plt.show()
```
![image](https://github.com/epiklife/Aiffel_Assignment/assets/131635437/b082c31b-ae50-4b2e-8d06-e28db3a1c32c)

- addWeighted를 이용하여 스티커 이미지를 원본이미지상에 출력하였다.

- [x] 주석을 보고 작성자의 코드가 이해되었나요?
```
for dlib_rect, landmark in zip(dlib_rects, list_landmarks): # 얼굴 영역을 저장하고 있는 값과 68개의 랜드마크를 저장하고 있는 값으로 반복문 실행
    print (landmark[33]) # 코끝의 index는 33
    x = landmark[33][0] # 이미지에서 코끝 부위의 x값
    y = landmark[33][1] # 이미지에서 코끝 부위의 y값
    w = h = dlib_rect.width() # 얼굴 영역의 가로를 차지하는 픽셀의 수
    print (f'(x,y) : ({x},{y})')
    print (f'(w,h) : ({w},{h})')
```
landmark의 인덱스 의미 등의 의미를 주석으로 설명하여 값이 어떤 의미인지 잘 이해할 수 있었다.

- [x] 코드가 에러를 유발할 가능성이 있나요?

-코드가 에러를 발생할 여지가 없어 보인다.

- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
```
model_path = 'shape_predictor_68_face_landmarks.dat'
    # 저장한 landmark 모델의 주소를 model_path 변수에 저장
landmark_predictor = dlib.shape_predictor(model_path)
list_landmarks = []
    # 랜드마크의 위치를 저장할 list 생성    

# 얼굴 영역 박스 마다 face landmark를 찾음
# face landmark 좌표를 저장
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
        # 모든 landmark의 위치정보를 points 변수에 저장
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        # 각각의 landmark 위치정보를 (x,y) 형태로 변환하여 list_points 리스트로 저장
    list_landmarks.append(list_points)
        # list_landmarks에 랜드마크 리스트를 저장

print(len(list_landmarks[0]))
    # 얼굴이 n개인 경우 list_landmarks는 n개의 원소를 갖고
    # 각 원소는 68개의 랜드마크 위치가 나열된 list 
    # list_landmarks의 원소가 1개이므로 list_landmarks[1]을 호출하면 IndexError가 발생
```
landmark predictor가 어떤 기능을 하고 어떤 결과를 출력하는지에 대해 충분히 이해하고 있다.

- [x] 코드가 간결한가요?
```
sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
# img_bgr은 rgb로만 적용해놓은 원본 이미지 
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    cv2.addWeighted(sticker_area, 0.5, np.where(img_sticker>127,sticker_area,img_sticker).astype(np.uint8), 0.5, 0)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) 
plt.show()
```
- 코드가 같은줄에서 길어지는 경우 가독성이 떨어질수 있는데 \를 사용하여 가독성을 증가시켰다
