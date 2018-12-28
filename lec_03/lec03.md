#LEC03 - Linear regression의 cost 최소화 알고리즘의 원리
========================================

1) Simplified Hypothesis : 간단한 설명을 위해 b를 생략

    H(x)=Wx
    
    Cost(W)=1/m Sigma i=1~m ( H(x.i) - y.i)^2
    
    => minimize 부분은 미분을 통해서 얻을수있다(2차 함수의 변곡점을 미분한 값이 0인 지점)
  --------------- --------------- --------------- --------------- ---------------
  
2) Gradient Descent algorithm : 경사가 0이 되는 지점을 찾는 알고리즘 

    W := W- learning_rage * a/aw * Cost(W) == 1 - learning_rate * 1/m Sigma i=1~m ( H(x.i) - y.i)^2

    a/aw * Cost(W) = gradient
    
    learning rate와 gradient를 곱한만큼 이동하며 학습하므로, 적절한 learning_rate를 주는게 중요하다
    너무 크게 값을 주면 학습이 발산하고, 너무 작게 값을 주면 학습하는데 있어 시간이 오래 걸린다.

    Cost function이 밥그릇을 엎은 모양일때만 Gradient descent algorithm 사용가능.
