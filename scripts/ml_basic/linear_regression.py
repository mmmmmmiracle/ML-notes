import numpy as np

class LR_GD():
	'''梯度下降'''
    def __init__(self):
        self.w = None     
    def fit(self,X,y,alpha=0.02,loss = 1e-10): # 设定步长为0.002,判断是否收敛的条件为1e-10
        y = y.reshape(-1,1) #重塑y值的维度以便矩阵运算
        [m,d] = np.shape(X) #自变量的维度
        self.w = np.zeros((d)) #将参数的初始值定为0
        tol = 1e5
        while tol > loss:
            h_f = X.dot(self.w).reshape(-1,1) 
            theta = self.w + alpha*np.mean(X*(y - h_f),axis=0) #计算迭代的参数值
            tol = np.sum(np.abs(theta - self.w))
            self.w = theta
    def predict(self, X):
        # 用已经拟合的参数值预测新自变量
        y_pred = X.dot(self.w)
        return y_pred 



class LR_LS:
	'''最小二乘法'''
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept # bias 截距

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.w = np.dot(pseudo_inverse, y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.w)


def first_derivativ(X, y, w):
    return np.matmul(X.T, np.matmul(X, w) - y)

def second_derivative(X):
    return np.matmul(X.T, X)

def get_error(X, y, w):
    tmp = np.matmul(X, w) - y
    return np.matmul(tmp.T, tmp) / 2
 
def get_min_m(X, y, sigma, delta, d, w, g):
    m = 0
    while True:
        w_new = w + pow(sigma, m) * d
        left = get_error(X, y , w_new)
        right = get_error(X, y , w) + delta * pow(sigma, m) * g.T * d
        if left <= right:
            break
        else:
            m += 1
    return m   

class LR_NEWTON():
	'''牛顿法求解'''
    def __init__(self):
        self.w = None
    
    def fit(self, X, y, max_iter=50, sigma=0.5, delta=0.25):
        '''
            sigma : (0, 1)
            delta : (0. 0.5)
        '''
        print(type(X))
        n = X.shape[1]
        self.w = np.mat(np.ones((n, 1)))
        it = 0
        while it <= max_iter:
            g = first_derivativ(X, y, self.w)
            G = second_derivative(X)
            try:
                d = -np.linalg.inv(G) * g
            except Exception:
                print('不可逆')
                break
            m = get_min_m(X, y, sigma, delta, d, self.w, g)  # 得到最小的m
            self.w = self.w + pow(sigma, m) * d
            if it % 10 == 9:
                print ("\t---- itration: ", it, " , error: ", get_error(X, y , self.w)[0, 0])
            it += 1
    def predict(self, X):
        y_pred = X.dot(self.w)
        return y_pred 

class RidgeRegression:
	'''加l2正则的岭回归'''
    def __init__(self, fit_intercept=True):
        self.w = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y, alpha=1):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        A = alpha * np.eye(X.shape[1])
#         A = 0
        pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X) + A), X.T)
        self.w = np.dot(pseudo_inverse, y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.w)