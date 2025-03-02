---
title: 텐서란 무엇인가, Matmul, mul 차이점
author: naturesh
date: 2025-03-02 22:35:00 +0800
categories: [MATH, AI]
tags: [matrix, tensor]
math: true
---



### 텐서란 무엇인가.

텐서는 쉽게말해서 n차원 배열이다. 1차원은 높이, 2차원은 가로, 3차원은 세로

공부할때 그린 이미지로 쉽게 이해해보자.

![학습할때 사용한 이미지](/assets/img/posts/what-is-tensor.jpg)


그렇다면 만약 NLP를 위한 3차원 텐서가 있다고 해보자. 

`1차원 : batch_size`

`2차원 : 맥락`

`3차원 : 문장` 

이라고 간단히 이해할수 있을것 같다. 즉 한 면이 문장들을 이루는 데이터셋이
되는것이다. 


### Pytorch 에서는 어떻게 차원을 확인할까?
```python

    ndimObject.dim() -> int  # n
    ndinObject.shape -> (n,n1,n2)


```


### Multiplication vs Matrix Multiplication

$$
\begin{bmatrix} 
1 & 2 \\
3 & 4 \\
\end{bmatrix}
와
\begin{bmatrix} 
1 \\
2 \\
\end{bmatrix}
$$

를 연산하고자 한다. 

기본적인 Multiplication은 broadcasting을 통해 

$$
\begin{bmatrix} 
1 \\
2 \\
\end{bmatrix}
=>
\begin{bmatrix} 
1 & 1 \\
2 & 2 \\
\end{bmatrix}
$$

로 변해 각 위치마다 연산 된다. 

하지만 

Matrix Multiplication 은 
`2x2` 행렬과 `2x1`의 행렬이 곱 되므로 `2x1`의 결과가 나와야 한다.

따라서 일반적인 행렬 곱셈으로 계산이 된다.

결과는 다음과 같다.


$$
Mul=
\begin{bmatrix} 
1 & 2 \\
6 & 8 \\
\end{bmatrix}
,
MatMul=
\begin{bmatrix} 
5 \\
11 \\
\end{bmatrix}
$$

`출처 : 모두를 위한 딥러닝 시즌2`
