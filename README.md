# MusicAI

DCGAN을 활용한 음악 생성 AI

이 모델의 아이디어는 음악을 이미지로 바꾸고 그 이미지를 이미지 생성 모델에 학습시킨 뒤, 그 학습된 모델의 출력물을 다시 소리로 바꾸면 음악이 된다. 여기에서는 이미지 생성 모델로 DCGAN을 사용하였다.


파일들의 역할
- Converter.py: 음악 파일을 Spectrogram으로 변환하거나 SPectrogram을 음악 파일, 사진으로 변환함.
- DataMaker.py: 특정 폴더에 들어있는 음악 파일들을 불러와 일정한 간격으로 자르고, 자른 음악들을 Converter.py에서 Spectrogram으로 변환한다.
- model.py: DataMaker.py에서 만든 데이터들을 불러와 모델을 학습하여 저장함. 저장된 모델을 불러와 음악을 생성한다.
- app.py: streamlit 라이브러리를 사용하여 웹사이트에서 음악 생성을 쉽게 할 수 있다.

Discriminator의 구조
- DCGAN 논문에서, Discriminator는 활성화 함수로 모두 LeakyReLU(기울기 0.2)를 사용하고, output layer에는 이진 분류를 위해 sigmoid를 사용한다. 과적합 방지를 위한 BatchNorm과 Dropout(50%)를 사용했다.

Discriminator의 순전파 과정
- conv2d를 거치며 입력 이미지 크기는 점차 줄어들고, 이미지의 채널 수는 점점 늘어난다.
- 연산량 감소를 위해 채널 수를 줄이는 1x1conv를 사용함. 이로써 어떤 특징이 더 중요한지 학습하여 새로운 특징 맵을 얻게 된다.
- 마지막 output layer에서는 linear layer와 sigmoid를 거쳐 0~1 사이의 값을 가지는 하나의 값으로 출력된다.

Generator의 구조
- DCGAN 논문에서, Generator는 활성화 함수로 모두 ReLU를 사용하고, output layer에는 hyperbolic tangent를 사용한다. 과적합 방지를 위해 BatchNorm을 사용하였다.

Generator의 순전파 과정
- 랜덤 벡터 z를 받아 linear layer를 거치며 출력 이미지보다 32배 작은 "초기 이미지"를 생성한다.
- 1x1convTranspose를 거치며 "초기 이미지"의 채널 수를 늘린다.
- convTranspose2d를 거치며 이미지 크기를 점차 늘리며 채널 수는 점차 줄인다.
- 마지막 output layer에서는 채널 수를 한 개로 만들고, Tanh를 거치며 각각 픽셀들의 값의 범위를 -1~1로 정규화한다.

