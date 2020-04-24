# 机器学习基础：MLE和EM算法

**假设一个情景**：假设某种实验有四个可能得结果，其发生概率分别为
$$
\frac{1}{2}-\frac{\theta}{4},\frac{1}{4}-\frac{\theta}{4},\frac{1}{4}+\frac{\theta}{4},\frac{\theta}{4}
$$
且次数分别为$y_1, y_2, y_3, y_4$， 求$\theta$的估计值。

## 1. MLE

$$
\begin{split}
L(\theta) &= (\frac{1}{2}-\frac{\theta}{4})^{y_1}(\frac{1}{4}-\frac{\theta}{4})^{y_2}(\frac{1}{4}+\frac{\theta}{4})^{y_3}(\frac{\theta}{4})^{y_4} \\

ln\,L(\theta) &= y_1ln\,\frac{2-\theta}{4} + 
y_2ln\,\frac{1-\theta}{4} + 
y_3ln\,\frac{1+\theta}{4} +
y_4ln\,\frac{\theta}{4}\\

\frac{dln\,L(\theta)}{d\theta} &= -\frac{y_1}{2-\theta} -\frac{y_2}{1-\theta} + \frac{y_3}{1+\theta}+\frac{y_4}{\theta} = 0
\end{split}
$$

上面假设的场景，实验结果都是可以直接观测的，此时可以使用MLE。但是如果实验结果含有隐藏变量，即不可观测部分，就需要用到EM算法。



在上面的情景中，假设第一部分$\frac{1}{2}-\frac{\theta}{4}$可以分为$\frac{1}{4}-\frac{\theta}{4}, \frac{1}{4}$，且出现次数分别为$z_1, y_1-z_1$。第三部分$\frac{1}{4}+\frac{\theta}{4}$可以分为$\frac{\theta}{4}，\frac{1}{4}$， 且出现次数分别为$z_2, y_3-z_2$。则：
$$
\begin{split}
L(\theta) &= (\frac{1}{4}-\frac{\theta}{4})^{z_1 + y_2}
(\frac{1}{4})^{y_1-z_1}
(\frac{\theta}{4})^{y_4+z_2}
(\frac{1}{4})^{y_3-z_2}\\

ln\,L(\theta) &= (z_1+y_2)ln\,\frac{1-\theta}{4} + 
(z_2 + y_4)ln\,\frac{\theta}{4} +
(y_1-z_1 + y_3-z_2)ln\,\frac{1}{4} \\

\frac{dln\,L(\theta)}{d\theta} &= -\frac{z_1 + y_2}{1-\theta} +\frac{z_2 + y_4}{\theta} = 0 \\

\hat{\theta} &= \frac{z_2+y_4}{z_1+z_2+y_2+y_4} \\

&其中z_1 \sim B(y_1, \frac{1-\theta}{2-\theta}), \, \, z_2 \sim B(y_3, \frac{\theta}{1+\theta})
\end{split}
$$

## 2. EM算法

1. E步,目的是消去潜在变量$z_1, z_2$:
   $$
   E(z_1) = \frac{1-\theta}{2-\theta}y_1, E(z_2) = \frac{\theta}{1+\theta}y_3
   $$
   带入得到：
   $$
   \hat{\theta} = \frac{\frac{\theta}{1+\theta}y_3+y_4}{\frac{1-\theta}{2-\theta}y_1+\frac{\theta}{1+\theta}y_3+y_2+y_4}
   $$

2. M步，取最大值

$$
\theta^{(i+1)} = \frac{\frac{\theta^{(i)}}{1+\theta^{(i)}}y_3+y_4}{\frac{1-\theta^{(i)}}{2-\theta^{(i)}}y_1+\frac{\theta^{(i)}}{1+\theta^{(i)}}y_3+y_2+y_4}
$$

任意取初始$\theta \in (0,1) $，不断迭代