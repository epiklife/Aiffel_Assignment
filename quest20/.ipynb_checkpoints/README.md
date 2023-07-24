## 아이펠캠퍼스 온라인4기 피어코드리뷰 [2023-07-24]

- 코더 : 김창완
- 리뷰어 : 최우정

----------------------------------------------


## [O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   

1. multiface detection을 위한 widerface 데이터셋의 전처리가 적절히 진행되었다.

 "source": [
    "전처리\n",
    
    "# ground truth txt와 image_file에서 image_info를 반환하는 함수 정의\n",
    "import os, cv2, time\n",
    "import tensorflow as tf\n",
    "import tqdm\n",
    "import numpy as np\n",
    "import math\n",
    "from itertools import product\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "PROJECT_PATH = os.getenv('HOME')+'/aiffel/face_detector'\n",
    "DATA_PATH = os.path.join(PROJECT_PATH, 'widerface')\n",
    "MODEL_PATH = os.path.join(PROJECT_PATH, 'checkpoints')\n",
    "TRAIN_TFRECORD_PATH = os.path.join(PROJECT_PATH, 'dataset', 'train_mask.tfrecord')\n",
    "VALID_TFRECORD_PATH = os.path.join(PROJECT_PATH, 'dataset', 'val_mask.tfrecord')\n",
    "CHECKPOINT_PATH = os.path.join(PROJECT_PATH, 'checkpoints')\n",
    "\n",
    "DATASET_LEN = 12880\n",
    "BATCH_SIZE = 32\n",
    "IMAGE_WIDTH = 320\n",
    "IMAGE_HEIGHT = 256\n",
    "IMAGE_LABELS = ['background', 'face']\n",
    "\n",
    "\n",
    "def parse_box(data):\n",
    "    x0 = int(data[0])\n",
    "    y0 = int(data[1])\n",
    "    w = int(data[2])\n",
    "    h = int(data[3])\n",
    "    return x0, y0, w, h\n",
    "\n",
    "def parse_widerface(file):\n",
    "    infos = []\n",
    "    with open(file) as fp:\n",
    "        line = fp.readline()\n",
    "        while line:\n",
    "            n_object = int(fp.readline())\n",
    "            boxes = []\n",
    "            for i in range(n_object):\n",
    "                box = fp.readline().split(' ')\n",
    "                x0, y0, w, h = parse_box(box)\n",
    "                if (w == 0) or (h == 0):\n",
    "                    continue\n",
    "                boxes.append([x0, y0, w, h])\n",
    "            if n_object == 0:\n",
    "                box = fp.readline().split(' ')\n",
    "                x0, y0, w, h = parse_box(box)\n",
    "                boxes.append([x0, y0, w, h])\n",
    "            infos.append((line.strip(), boxes))\n",
    "            line = fp.readline()\n",
    "    return infos\n",
    "\n",


2. SSD 모델이 안정적으로 학습되어 multiface detection이 가능해졌다.
   
3. 이미지 속 다수의 얼굴에 스티커가 적용되었다.

"source": [
    "inference_test(FILE_PATH, TEST_IMAGE_PATH, STICKER_PATH, is_bbox=False, is_sticker=True)"
   ]

## [O] 주석을 보고 작성자의 코드가 이해되었나요?

주석 및 순서대로 코드가 잘 기재 되어 있어서 이해하기 편했습니다.  


## [X] 코드가 에러를 유발할 가능성이 있나요?

코드 자체에는 에러를 유발할 가능성이 보이지 않습니다.


## [O] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

순서적으로 코드를 잘 작성하셨기 때문에, 추가로 보완할 부분은 없다고 생각이 들었습니다. 

## [O] 코드가 간결한가요?
  
Default boxes 코드를 구현한 부분도 그렇고 전반적으로 코드를 작성하신 부분이 간결하여 좋았습니다. 


----------------------------------------------
