#LEC02 - Linear regression의 hypothesis 와 cost
-----------------------------

1) Supervised Leanring / Linear regression 예제
  ex) 공부시간에 따른 점수(0~100) 예측
    - 과거 실제 공부 시간에 따른 score date set 필요
    - 공부시간(x), 점수(H(x))의 그래프를 그려서 분포에 가장 적절한 Linear한 모델을 가설
      - H(x) = Wx + b
    - W,b 값을 임시로 주고 실제 데이터를 연산하여 H(x)의 값과 실제의 값의 차이인 Cost 계산
    - Cost가 최소가 되는 W,b 값을 찾기

=============

2) Linear Hypothesis
  - data의 규칙을 예상한 가설함수
    - H(x) = Wx + b
    
=============

3) Cost Function(=loss function)
  - hypothesis 로 계산한 H(x) 값과 실제로 주어진 y값과의 편차
  
=============

4) Linear regression 학습의 목표
  -Cost function이 가장 작은 W,b 값을 구하는것 ,= minimize(W,b)
