<p align="center">
  <img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/686699ac-bf3d-498e-ada1-b248887b8839" height="100px"/>
</p>   

<p align="center" style="font-size:200%"><b><i>fruit360-project</i></b></p>   

# 🍇과일이미지분류 프로젝트(fruit-360)
> Kaggle의 오픈데이터 활용 https://www.kaggle.com/datasets/moltean/fruits   
> 약 131개의 과일이미지를 분류하여 어떤 과일인지 예측하는 모델 프로젝트   
> 이미지를 학습시켜 분류하는 기계학습 프로젝트
## 🍉fruit-360 데이터 설명
<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/84724404-6317-478f-828b-ff09b4666596" width="40%"/>   
<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/244204bd-4d52-460c-a2b8-ea915878158a" width="45%"/>

> 67692개의 train data, 22688 test data구성, 100X100 픽셀 .jpg형식 이미지 데이터   
> 위 사진과 같이 한 과일에 대해 여러장의 이미지들이 들어있다. 이 이미지를 모델에 학습시켜 이미지를 분류   
> 오른쪽 사진은 과일의 종류를 나타낸 이미지


## 1. CNN모델 활용   
<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/3775d591-2a0c-4262-9195-bf8e96dfe289" width="45%"/>
<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/967f6b1d-0068-4dfe-97d8-fc1d6b0b0cce" width="45%"/>      

<b>해당 모델은 기존 CNN모델에 아래의 6가지를 적용하여 조정한 모델이다.</b>   

1. 기존 CNN모델에 배치 정규화를 진행. 배치 정규화는 각 미니 배치에 대한 레이어의 입력을 표준화하여 깊은 신경망을 훈련시킬 수 있는 기술이다. 학습 프로세스를 안정화하고 심층 네트워크를 훈련하는데에 필요한 훈련 에포크 수를 크게 줄이는 효과를 준다.         
2. 데이터 증대를 진행. 기존 데이터에 약간 수정된 사본을 추가하거나 기존 데이터에서 새로 생성된 합성 데이터를 추가하여 데이터 양을 늘리는 데 사용되는 기술이다. 정규화 장치 역할을 하며 기계학습 모델을 학습할 때 과적합을 줄이는 데 도움이 되어 적용하였다.      
3. 학습률 스케줄링을 진행. 이는 미리 정의된 스케줄에 따라 학습률을 줄여 모델을 학습시키며 학습중에 학습률을 조정하면서 진행하는 것을 말한다. 일반적인 학습 속도 스케줄링에는 시간 기반 감쇠, 단계 감쇠 및 지수 감쇠가 포함되어 있다.      
4. 가중치감쇠를 진행. 이는 가중치를 작게 유지하고 그래디언트 폭발을 방지한다. 가중치의 L2 표준이 loss에 추가되기 때문에 각 층의 반복은 손실 외에도 모델 가중치를 최적화/최소화하게 된다. 이렇게 하면 가중치를 가능한 한 작게 유지하여 가중치가 통제 불능 상태로 커지는 것을 방지하고 그래디언트 폭발을 방지할 수 있게 된다.      
5. 그래디언트 클래핑을 사용. 이를 사용하면 마찬가지로 신경망에서 그래디언트 폭발 방지가 가능하다. 그래디언트 클리핑은 그래디언트의 크기를 제한한다. 그래디언트 클리핑 계산 방법에는 여러 가지가 있지만 일반적인 방법은 그래디언트의 크기를 재조정하여 규범이 특정 값이 되도록 하는 방법이다.      
6. adaptive opitimizer적용. 경사 하강 알고리즘의 문제를 해결하기 위해 도입되었다. 가장 중요한 기능은 학습률을 조정할 필요가 없다는 것이다.      

> 오른쪽 이미지는 신경망 layer 구성을 나타냄   
> 각 신경망층을 sequential함수를 통해 연결, batchNorm2d를 사용하여 각 층의 끝에 배치 정규화 진행.   
> conv층은 총 3개   


<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/85bf703a-0cb0-4aaa-975e-80ff99757030" width="45%"/>
<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/81dae76e-e55f-420d-b2bb-ff0ae6176f76" width="45%"/>

> 모델학습결과 validation 정확도가 98%정도로 매우 높은 정확도가 나왔다.   
> 오른쪽 이미지는 학습시킨 모델로 임의의 이미지를 분류해 보았을때 정확한 결과가 나오는 것을 나타낸 것이다.

## 2. SVM모델 활용
SVM모델은 보통 이중분류가 가능하지만 모델을 수정하면 다소 비효율적일 순 있지만 다중분류가 가능하다.  


<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/eed570ff-ae59-4b53-8405-c79a16b639bd" width="30%"/>
<img src="https://github.com/ohdeng02/fruit360-project/assets/90545561/baeaa1be-a448-4bc5-9560-efdb47fd7c29" width="45%"/>

> 왼쪽 이미지는 ovr전략의 svm다중분류모델을 구성한 모습니다.   
> 그 결과 99%정도로 높은 정확도가 도출되었다.

## 결론
- 비교적 분류가 쉬운 데이터집합이라 정확도가 높게 나온 것으로 예상된다. 
- SVM분류모델이 정확도가 높게 나오긴 했지만 다중분류에는 비효율적이라고 생각되어 신경망구조인 CNN모델이 적합하다고 생각한다. 
