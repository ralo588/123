import numpy as np

def computeCost(X, y, theta):
    m = len(y)

    h=np.matmul(X, theta)

    for i in range(m):  # 500001초과는 500001로 고정
        if h[i] > 500001:
            h[i] = 500001

    Sum = sum((h-y)**2)   #sum(h(x)-y)^2

    J = Sum / (2 * m)

    return J
