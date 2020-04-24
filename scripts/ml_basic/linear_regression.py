import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error		

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
	return np.dot(X.T, np.dot(X, w) - y)

def second_derivative(X):
	return np.dot(X.T, X)

def get_error(X, y, w):
	tmp = np.dot(X, w) - y
	return np.dot(tmp.T, tmp) / 2
 
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
				print ("\t---- itration: ", it, " , error: ", get_error(X, y , self.w))
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
#		 A = 0
		pseudo_inverse = np.dot(np.linalg.inv(np.dot(X.T, X) + A), X.T)
		self.w = np.dot(pseudo_inverse, y)

	def predict(self, X):
		if self.fit_intercept:
			X = np.c_[np.ones(X.shape[0]), X]
		return np.dot(X, self.w)


if __name__ == "__main__":
	df = pd.read_csv('../../inputs/housing.csv')
	label = df[df.columns.values[-1]]
	print(df.columns)
	feature = df[df.columns[:-1]]
	train_x, test_x, train_y, test_y = train_test_split(feature.values, label.values.reshape(-1,1), shuffle=True, random_state=2020, test_size=0.2)
	lr_nt = LR_NEWTON()
	# lr_nt = RidgeRegression()
	# lr_nt = LR_LS()
	print(train_x.shape, train_y.shape)
	# lr_nt = LR_GD()
	lr_nt.fit(train_x, train_y)
	print(lr_nt.w)
	print(mean_squared_error(train_y, lr_nt.predict(train_x)), r2_score(train_y, lr_nt.predict(train_x)))
	print(mean_squared_error(test_y, lr_nt.predict(test_x)), r2_score(test_y, lr_nt.predict(test_x)))


