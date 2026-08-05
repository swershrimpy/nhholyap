import numpy as np
from sklearn.linear_model import LinearRegression

x_true = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 
          0.3, 0.6, 0.9, 1.2, 1.5, 1.8,
          0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
          0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
x_true = x_true.reshape((-1, 1))
x_odom = np.array([0.099345, 0.1965, 0.3073, 0.4175, 0.51753, 0.6201, 
          0.18673, 0.4442, 0.69476, 0.94, 1.195688, 1.44233,
          0.1189, 0.2905, 0.45389, 0.6169, 0.762256, 0.922953,
          0.15095, 0.3697, 0.57142, 0.7802, 0.9922, 1.187963])
x_odom = x_odom.reshape((-1, 1))

reg = LinearRegression().fit(x_odom, x_true)
print(reg.score(x_odom, x_true))

print(reg.coef_)
print(reg.intercept_)

# yaw_true = np.array([-0.3, -0.6, -1.5, -0.9, ])
# yaw_true = yaw_true.reshape((-1, 1))
# yaw_odom = np.array([-0.1896875, -0.256, -0.768, -0.8543, ])
# yaw_odom = yaw_odom.reshape((-1, 1))

# reg = LinearRegression().fit(x_odom, x_true)
# print(reg.score(x_odom, x_true))

# print(reg.coef_)
# print(reg.intercept_)