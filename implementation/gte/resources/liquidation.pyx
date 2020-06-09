import sys
import os
path_gte=os.path.abspath('./')
n=0
while (not os.path.basename(path_gte)=='gte') and (n<4):
    path_gte=os.path.dirname(path_gte)
    n+=1
if not os.path.basename(path_gte)=='gte':
    print("path_ gte not found. Instead: {}".format(path_gte))
    raise ValueError()
path_resources=path_gte+'/resources'
sys.path.append(path_gte+'/')
sys.path.append(path_resources+'/')
from priceprocess import Price
import numpy as np
cimport numpy as np
import pandas as pd
import bisect
import copy
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport exp, sqrt
import time
import matplotlib.pyplot as plt

DTYPEi=np.int
DTYPEf=np.float
DTYPEc=np.cdouble
ctypedef np.int_t DTYPEi_t
ctypedef np.float_t DTYPEf_t
ctypedef np.cdouble_t DTYPEc_t
cdef struct Coef:
    DTYPEf_t c1
    DTYPEf_t c2
    DTYPEf_t c3
cdef struct DiffEqData:
    DTYPEf_t q0
    DTYPEf_t qT
    DTYPEf_t r0
    DTYPEf_t S0
    DTYPEf_t terminal_time
    DTYPEf_t Ktilde

class Liquidator:
    def __init__(self,initial_inventory=None, horizon=None, impact_coef=None, risk_coef=None):
        cdef DiffEqData ODEdata
        self.ODEdata=ODEdata
        if initial_inventory!=None:
            self.set_initial_inventory(initial_inventory)
        if horizon!=None:
            self.set_horizon(horizon)
        if (impact_coef!=None) and (risk_coef!=None):
            self.set_coef(impact_coef,risk_coef)
    def set_initial_inventory(self, DTYPEf_t x):
        self.initial_inventory = x
        self.ODEdata['q0']=x
    def set_initial_price(self, DTYPEf_t S0):
        self.initial_price = S0
        self.ODEdata['S0']=S0
    def set_horizon(self, DTYPEf_t horizon):
        assert horizon>0.0
        self.horizon=horizon
    def set_coef(self, DTYPEf_t impact_coef, DTYPEf_t risk_coef):
        assert impact_coef>0.0
        cdef DTYPEf_t ratio = risk_coef/impact_coef
        cdef Coef coefficients
        coefficients.c1=impact_coef
        coefficients.c2=risk_coef
        coefficients.c3=ratio
        self.coefficients=coefficients
    def set_priceprocess(self,priceprocess):
        self.priceprocess=copy.copy(priceprocess)
    def compute_initial_rate(self,np.ndarray[DTYPEf_t, ndim=2] expectedprice, DTYPEf_t liquidation_target=0.0, DTYPEf_t t1=-1.0):
        if t1<0.0:
            t1=self.horizon
        self.ODEdata['terminal_time']=t1
        self.ODEdata['qT']=liquidation_target
        cdef DTYPEf_t c1tilde=2*self.coefficients['c1']**2, c3=self.coefficients['c3']
        cdef DTYPEf_t Ktilde=self.compute_Ktilde(expectedprice, t1)
        cdef DTYPEf_t res = (self.ODEdata['qT']-self.ODEdata['q0'])*np.cosh(c3*t1) + Ktilde
        res*=c3/np.sinh(c3*t1)
        res-=self.ODEdata['S0']/c1tilde
        self.ODEdata['r0']=res
    def compute_Ktilde(self,np.ndarray[DTYPEf_t, ndim=2] expectedprice, DTYPEf_t t1):
        assert t1<=self.horizon
        idx=expectedprice[:,0]<=t1
        cdef np.ndarray[DTYPEf_t, ndim=1] forecast=expectedprice[idx,1]
        cdef np.ndarray[DTYPEf_t, ndim=1] grid=expectedprice[idx,0]
        cdef np.ndarray[DTYPEf_t, ndim=1] dt=np.diff(grid, append=grid[len(grid)-1])
        cdef DTYPEf_t c1tilde=2*self.coefficients['c1']**2, c3=self.coefficients['c3']
        cdef DTYPEf_t res=(np.cosh(c3*t1)/c1tilde)\
                *np.sum(np.cosh(c3*grid)*forecast*dt)
        res-=(np.sinh(c3*t1)/c1tilde)\
                *np.sum(np.sinh(c3*grid)*forecast*dt)
        self.ODEdata['Ktilde']=res
        return res
    def compute_inventory_trajectory(self,type_of_price='simulation', DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0):
        if type_of_price=='simulation':
            priceinfo=np.array(self.priceprocess.simulation, copy=True)
            forecast=np.array(self.priceprocess.expected_trajectory, copy=True)
        elif type_of_price=='data':
            priceinfo=np.array(self.priceprocess.data, copy=True)
            forecast=np.array(self.priceprocess.fourier.series_val, copy=True)
            forecast[:,1]=np.exp(forecast[:,1])
        if t1<=t0:
            t0=priceinfo[0,0]
            t1=self.horizon
        idx=(priceinfo[:,0]>=t0)
        cdef np.ndarray[DTYPEf_t, ndim=2] price = np.array(priceinfo[idx,:], copy=True)
        self.set_initial_price(price[0,1])
        idx=(forecast[:,0]>=t0)
        cdef np.ndarray[DTYPEf_t, ndim=2] expectedprice = np.array(forecast[idx,:],copy=True)
        self.compute_initial_rate(expectedprice, t1=t1)
        cdef np.ndarray[DTYPEf_t, ndim=2] traj = np.ones((len(price),3),dtype=DTYPEf)
        traj[:,0]=np.array(price[:,0], copy=True)
        traj[0,1]=self.ODEdata['q0']
        traj[0,2]=self.ODEdata['r0']
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = np.diff(price[:,0])
        cdef np.ndarray[DTYPEf_t, ndim=1] dS = np.diff(price[:,1])
        cdef DTYPEf_t qT = self.ODEdata['qT']
        cdef DTYPEf_t c1tilde=2*self.coefficients['c1']**2, c3sq=self.coefficients['c3']**2
        cdef int n=0
        while n<len(dt) and traj[n,1]>qT:
            traj[n+1,1]=traj[n,1]+traj[n,2]*dt[n]
            traj[n+1,2]=traj[n,2]\
                    +c3sq*(traj[n,1]-qT)*dt[n] - dS[n]/c1tilde
            n+=1
        cdef np.ndarray[DTYPEf_t, ndim=2] res = np.array(traj[:n,:], copy=True)
        liquidation={'price':price, 'expected_price':expectedprice, 'inventory_trajectory': res}
        self.liquidation=liquidation
    def plot(self, DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0, show_price=True):
        if t1<=t0:
            t0=self.liquidation['inventory_trajectory'][0,0]
            t1=self.horizon
        idx=np.logical_and(self.liquidation['price'][:,0]>=t0,
                           self.liquidation['price'][:,0]<=t1)
        price=np.array(self.liquidation['price'][idx,:], copy=True)
        idx=np.logical_and(self.liquidation['expected_price'][:,0]>=t0,
                           self.liquidation['expected_price'][:,0]<=t1)
        exprice=np.array(self.liquidation['expected_price'][idx,:], copy=True)
        idx=np.logical_and(self.liquidation['inventory_trajectory'][:,0]>=t0,
                           self.liquidation['inventory_trajectory'][:,0]<=t1)
        traj=np.array(self.liquidation['inventory_trajectory'][idx,:], copy=True)
        fig=plt.figure(figsize=(15,12))
        if show_price:
            axp=fig.add_subplot(211)
            axp.plot(price[:,0], price[:,1], label='price', color='blue')
            axp.plot(exprice[:,0], exprice[:,1], label='reversion_target', color='lightblue', linestyle='--', linewidth=3)
            axp.set_xlabel('time')
            axp.set_ylabel('price')
            axi=axp.twinx()
            axi.plot(traj[:,0], traj[:,1], label='inventory', color='darkred')
            axi.set_ylabel('inventory')
            axp.legend()
            axi.legend()
            axp=fig.add_subplot(212)
            axp.plot(price[:,0], price[:,1], label='price', color='blue')
            axp.plot(exprice[:,0], exprice[:,1], label='reversion_target', color='lightblue', linestyle='--', linewidth=3)
            axp.set_xlabel('time')
            axp.set_ylabel('price')
            axi=axp.twinx()
            axi.plot(traj[:,0], traj[:,2], label='inventory_rate', color='darkred')
            axi.set_ylabel('rate')
            axp.legend()
            axi.legend()
        else:
            axi=fig.add_subplot(211)
            axi.plot(traj[:,0], traj[:,1], label='inventory', color='darkred')
            axi.set_xlabel('time')
            axi.set_ylabel('inventory')
            axi=fig.add_subplot(212)
            axi.plot(traj[:,0], traj[:,2], label='inventory_rate', color='darkred')
            axi.set_xlabel('time')
            axi.set_ylabel('rate')
            axi.legend()
            axi.legend()
        plt.show()
        

            







        


        
        
