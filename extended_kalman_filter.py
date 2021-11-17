#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt


# In[ ]:


#X:[x, y, yaw, v]
#z:[x, y]
#u:[v, yawrate]

class EKF:
    def __init__(self):
        # Estimation parameter of EKF
        self.Q = np.diag([0.1, 0.1, math.radians(1.0), 1.0])**2 #システムノイズ
        self.R = np.diag([1.0, math.radians(40.0)])**2 #センサーノイズ
        self.X_est = np.matrix(np.zeros((4, 1))) #システムの状態推定値
        self.P_est = np.eye(4) #誤差の共分散行列
        
    def setInput(self, v, yawrate):
        #制御入力
        u = np.matrix([v, yawrate]).T
        return u
    
    def setSensor(self, x, y):
        #観測値
        z = np.matrix([x, y])
        return z
    
    #システムの時間遷移に関する線形モデル
    def calcState(self, X, u, dt):
        F = np.matrix([[1.0, 0, 0, 0],
                       [0, 1.0, 0, 0],
                       [0, 0, 1.0, 0],
                       [0, 0, 0, 0]])

        B = np.matrix([[dt * math.cos(X[2, 0]), 0],
                       [dt * math.sin(X[2, 0]), 0],
                       [0.0, dt],
                       [1.0, 0.0]])
        #状態
        X = F * X + B * u
        return X


    def calcObservation(self, X):
        #  Observation Model
        H = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        z = H * X

        return z


    def jacobF(self, X, u, dt):
        yaw = X[2, 0]
        v = u[0, 0]
        jF = np.matrix([
            [1.0, 0.0, -dt * v * math.sin(yaw), dt * math.cos(yaw)],
            [0.0, 1.0, dt * v * math.cos(yaw), dt * math.sin(yaw)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        return jF


    def jacobH(self):
        # Jacobian of Observation Model
        jH = np.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return jH


    def estimate(self, z, u, dt):
        #  Predict
        X_pred = self.calcState(self.X_est, u, dt) #前回推定値から、dt進める
        jF = self.jacobF(X_pred, u, dt)
        P_pred = jF * self.P_est * jF.T + self.Q
        #  Update
        jH = self.jacobH()
        z_pred = self.calcObservation(X_pred)
        y = z.T - z_pred
        S = jH * P_pred * jH.T + self.R
        K = P_pred * jH.T * np.linalg.inv(S)
        self.X_est = X_pred + K * y
        self.P_est = (np.eye(len(self.X_est)) - K * jH) * P_pred


# In[ ]:


def main():
    ekf = EKF()
    print(ekf.X_est)
    u = ekf.setInput(1.0, 0.0)# v[m/s], yawrate[rad/s]
    z = ekf.setSensor(1.0, 0.0)# x[m], y[m]
    dt = 1.0
    ekf.estimate(z, u, dt)
    print(ekf.X_est)
    u = ekf.setInput(1.0, 0.0)# v[m/s], yawrate[rad/s]
    z = ekf.setSensor(2.0, 0.0)# x[m], y[m]
    ekf.estimate(z, u, dt)
    print(ekf.X_est)
    u = ekf.setInput(1.0, 0.0)# v[m/s], yawrate[rad/s]
    z = ekf.setSensor(3.0, 0.0)# x[m], y[m]
    ekf.estimate(z, u, dt)
    print(ekf.X_est)


# In[ ]:




if __name__ == '__main__':
    main()

