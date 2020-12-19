import numpy as np
import computeCost as cc


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = [0 for i in range(num_iters)]
    theta_len = len(theta)  #feature의 수
    iter=0

    for iter in range(num_iters):

        if iter >= 3:
            if abs((J_history[iter - 2] - J_history[iter - 1])) < 0.00001:
                break

        Sum = [0 for i in range(theta_len)]
        temp = np.array(np.zeros(theta_len))   #동시 업데이트를 위한 임시 저장 변수

        for i in range(theta_len):
            Sum[i] = sum((np.matmul(X, theta)-y)*np.array(X[:,i]))   #sum(h(x)-y)

        for i in range(theta_len):
            temp[i] = theta[i]-((Sum[i] / m) * alpha)

        theta = temp  #simultaneously update for every j=0,....,n

        J_history[iter] = cc.computeCost(X, y, theta)

    return theta, J_history[iter]