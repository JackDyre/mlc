# mlc

$$C = \left(a_{-1}-y\right)^2$$
$$a_L = \sigma\left(z_L\right)$$
$$z_L = a_{L-1}w_{L-1} + b_{L-1}$$

$$\frac{\partial C}{\partial a_{-1}}=2\left(a_{-1}-y\right)$$
$$\frac{\partial C}{\partial w_L}=\sum\frac{\partial C}{\partial _{L+1}}\frac{\partial a_{L+1}}{z_{L+1}}\frac{z_{L+1}}{w_{L+1}}=\sum w_L\sigma^\prime(z_L)\frac{\partial C}{\partial _{L+1}}$$
$$\frac{\partial C}{\partial w_L}=\frac{\partial C}{\partial a_{L+1}}\frac{\partial a_{L+1}}{z_{L+1}}\frac{z_{L+1}}{w_{L+1}}=a_L\sigma^\prime(z_L)\frac{\partial C}{\partial a_{L+1}}$$
$$\frac{\partial C}{\partial b_L}=\frac{\partial C}{\partial a_{L+1}}\frac{\partial a_{L+1}}{z_{L+1}}\frac{z_{L+1}}{b_{L+1}}=\sigma^\prime(z_L)\frac{\partial C}{\partial a_{L+1}}$$