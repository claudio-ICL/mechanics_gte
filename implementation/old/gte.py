#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import pandas as pd

import scipy

from scipy.stats import norm as normalDistribution

from scipy.stats import expon as exponentialDistribution

from scipy.stats import pareto 

from scipy.stats import bernoulli

from scipy.stats import cauchy as cauchyDistribution

from scipy.stats import t as tStudent

from fbm import FBM

import matplotlib.pyplot as plt

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import normalize

from mpl_toolkits.mplot3d import Axes3D 

from termcolor import colored

from scipy import optimize

from numpy.polynomial.polynomial import*


# In[2]:


import times_

from times_ import randomSign


# In[3]:


class liquidation(object):
    def __init__(self,
                 pricePath,
                 expectedPricePath,
                 initialInventory=100.0,
                 liquidationTarget=0.0,
                 coeffMarketImpact=1,
                 coeffRiskAversion=1,
                 initialTime=0,
                 timeHorizon=1,
                 method_extensionOfInventoryTrajectory='constant rate',
                 compute_costFunctional=True,
                 compute_revenues=True):
        
        self.pricePath=pd.Series(pricePath,copy=True)
        self.expectedPricePath=pd.Series(expectedPricePath,copy=True)
        time=np.array(pricePath.index,copy=True)
        if not (np.all(time==self.expectedPricePath.index)):
            print(colored('inventoryTrajectory: WARNING: (np.all(self.pricePath.index==self.expectedPricePath.index)={}'
                          .format(np.all(self.pricePath.index==self.expectedPricePath.index)),'red'))
        if not (time[-1]>=timeHorizon):
            print(colored('inventoryTrajectory: WARNING: (self.pricePath.index[-1]>=timeHorizon)={}'
                          .format((self.pricePath.index[-1]>=timeHorizon)),'red'))
        self.initialInventory=initialInventory
        self.liquidationTarget=liquidationTarget
        self.coeffMarketImpact=coeffMarketImpact
        self.coeffRiskAversion=coeffRiskAversion
        self.initialTime=initialTime
        self.timeHorizon=timeHorizon
        (core_inventoryTrajectory,core_executionRate)=(
            self.simulate_EulerLagrangeEquation_LTI(self.pricePath.loc[self.initialTime:self.timeHorizon].values,
            self.expectedPricePath.loc[self.initialTime:self.timeHorizon].values,
            initialInventory=self.initialInventory,
            liquidationTarget=self.liquidationTarget,
            coeffMarketImpact=self.coeffMarketImpact,
            coeffRiskAversion=self.coeffRiskAversion,
            initialTime=self.initialTime,
            timeHorizon=self.timeHorizon))
        self.errorOfLiquidation=self.compute_errorOfLiquidation_and_terminationTime(core_inventoryTrajectory,
                                                                                    liquidationTarget=self.liquidationTarget)
        (trajectory,executionRate,self.terminationTime)=self.handle_termination(core_inventoryTrajectory,
                                                       core_executionRate,
                                                       method_extensionOfInventoryTrajectory=method_extensionOfInventoryTrajectory)
        
        self.core_inventoryTrajectory=pd.Series(core_inventoryTrajectory,
                                                index=self.pricePath.loc[:self.timeHorizon].index)
        self.core_executionRate=pd.Series(core_executionRate,
                                                index=self.pricePath.loc[:self.timeHorizon].index)
        
        self.trajectory=pd.Series(trajectory,
                                  index=self.pricePath.loc[:np.maximum(self.timeHorizon,self.terminationTime)].index)
        
        self.executionRate=pd.Series(executionRate,
                                     index=self.pricePath.loc[:np.maximum(self.timeHorizon,self.terminationTime)].index)
        
        self.xi=self.compute_xi(self.pricePath,
                                self.expectedPricePath,
                                coeffMarketImpact=self.coeffMarketImpact,
                                coeffRiskAversion=self.coeffRiskAversion,
                                timeHorizon=self.timeHorizon)
        
        
#         (static_inventoryTrajectory,static_executionRate)=self.compute_staticInventory()
        
        
#         Alternatively, static_inventory is computed via Euler-Lagrange equation where the price path is taken to be the expeced price path.
        (static_inventoryTrajectory,static_executionRate)=self.simulate_EulerLagrangeEquation_LTI(
            self.expectedPricePath.loc[self.initialTime:self.timeHorizon].values,
            self.expectedPricePath.loc[self.initialTime:self.timeHorizon].values,
            initialInventory=self.initialInventory,
            liquidationTarget=self.liquidationTarget,
            coeffMarketImpact=self.coeffMarketImpact,
            coeffRiskAversion=self.coeffRiskAversion,
            initialTime=self.initialTime,
            timeHorizon=self.timeHorizon)

        self.static_inventoryTrajectory=pd.Series(static_inventoryTrajectory,
                                     index=self.expectedPricePath.loc[:self.timeHorizon].index)
        self.static_executionRate=pd.Series(static_executionRate,
                                     index=self.expectedPricePath.loc[:self.timeHorizon].index)
        
        if (compute_costFunctional):
            self.distance_coreExecution_staticExecution=self.compute_normInventoryTrajectory(
                self.core_inventoryTrajectory-self.static_inventoryTrajectory,
                self.core_executionRate-self.static_executionRate,
                coeffMarketImpact=self.coeffMarketImpact,
                coeffRiskAversion=self.coeffRiskAversion)
            distance_finalInventories_coreAndStatic=np.abs(
                self.core_inventoryTrajectory.loc[self.timeHorizon]
                -self.static_inventoryTrajectory.loc[self.timeHorizon])
            test_staticInPathwiseNeighbourhood=(-distance_finalInventories_coreAndStatic
                +self.xi*(self.distance_coreExecution_staticExecution**2))
            self.is_StaticInPathwiseNeighbourhood=(test_staticInPathwiseNeighbourhood>=0)
            self.costFunctional_coreExecution=self.compute_riskAdjustedCostOfExecution(
                self.core_inventoryTrajectory,self.core_executionRate)
            self.costFunctional_static=self.compute_riskAdjustedCostOfExecution(
                self.static_inventoryTrajectory,self.static_executionRate)
            self.costFunctional_inventory=self.compute_riskAdjustedCostOfExecution(self.trajectory,self.executionRate)
            self.is_costInventoryLowerThanCostStatic=(self.costFunctional_inventory<=self.costFunctional_static )
            self.is_costCoreLowerThanCostStatic=(self.costFunctional_coreExecution <=self.costFunctional_static)
        if (compute_revenues):
            self.revenue_static=self.compute_revenuesFromTrade(self.static_executionRate)
            self.revenue_core=self.compute_revenuesFromTrade(self.core_executionRate)
            self.revenue_trajectory=self.compute_revenuesFromTrade(self.executionRate)
            self.is_revenueCoreHigherThanStatic=(self.revenue_core>=self.revenue_static)
            self.is_revenueTrajectoryHigherThanStatic=(self.revenue_trajectory>=self.revenue_static)
        
#         print('liquidation: self.distance_coreExecution_staticExecution={}'.format(self.distance_coreExecution_staticExecution))
#         print('liquidation: test_staticInPathwiseNeighbourhood={}'.format(test_staticInPathwiseNeighbourhood))
#         print('liquidation: self.is_costInventoryLowerThanCostStatic={}'.format(self.is_costInventoryLowerThanCostStatic))
#         print('liquidation: self.is_costCoreLowerThanCostStatic={}'.format(self.is_costCoreLowerThanCostStatic))
#         print('liquidation: self.is_revenueCoreHigherThanStatic={}'.format(self.is_revenueCoreHigherThanStatic))
#         print('liquidation: self.is_revenueTrajectoryHigherThanStatic={}'.format(self.is_revenueTrajectoryHigherThanStatic))
        
    
    def compute_staticInventory(self):
        t=self.expectedPricePath.loc[self.initialTime:self.timeHorizon].index
        dt=np.diff(np.concatenate(([0],t)) )
        c_1=self.coeffMarketImpact
        c_3=self.coeffRiskAversion/self.coeffMarketImpact
        expected_price=np.array(self.expectedPricePath.loc[self.initialTime:self.timeHorizon].values,copy=True)
        if not (t.shape[0]==expected_price.shape[0]):
            print('compute_staticInventory: error: (t.shape[0]==expected_price.shape[0])={}'.format((t.shape[0]==expected_price.shape[0])))
        T=self.timeHorizon
        K=np.sum(np.cosh(c_3*(T-t))*expected_price*dt )/(2*np.square(c_1)*np.sinh(c_3*T))
        alpha_t=1-np.sinh(c_3*(T-t))/np.sinh(c_3*T)
        integral_t=np.zeros_like(t)
        for i in np.arange(1,t.shape[0]):
            time=t[i]-t[:i]
            if not (
                (np.logical_and(
                    np.all(time.shape==expected_price[:i].shape),
                    np.all(time.shape==dt[:i].shape)
                )
                )):
                print('compute_staticInventory: shape mismatch')
                print('time.shape={}'.format(time.shape))
                print('expected_price[:i].shape={}'.format(expected_price[:i].shape))
                print('dt[:i].shape={}'.format(dt[:i].shape))
                
            integral_t[i]=np.sum(
                np.cosh(c_3*time)*expected_price[:i]*dt[:i]
            )

            
        static_inventory=(
            (1-alpha_t)*self.initialInventory + alpha_t*self.liquidationTarget
            -integral_t/(2*np.square(c_1))
            +K*np.sinh(c_3*t)
        ) 
            
        static_rate=np.concatenate(
            (np.diff(static_inventory)/np.diff(t),
             [(np.diff(static_inventory)[-1])/(np.diff(t)[-1])]
            ))
        return static_inventory,static_rate
        
    
    def simulate_EulerLagrangeEquation_LTI(self,
                                           pricePath,
                                           expectedPricePath,
                                           initialInventory=100,
                                           liquidationTarget=0,
                                           coeffMarketImpact=1,
                                           coeffRiskAversion=1,
                                           initialTime=0,
                                           timeHorizon=1,
                                           numberOfPartitionPoints=1000,compute_r0=True,r_0=-1):
#             #pricePath is expected to be a numpy array, not a pandas series
        priceIncrement=np.diff(pricePath)
        ratioAversionOverImpact=coeffRiskAversion/coeffMarketImpact
        timePartition=self.pricePath.loc[initialTime:timeHorizon].index
        t=np.array(timePartition,copy=True)
        timeIncrement=np.diff(timePartition)
        dt=np.array(timeIncrement,copy=True)
        initialPrice=pricePath[0]
        Kappa_tilde=((np.cosh(ratioAversionOverImpact*timeHorizon)/(2*coeffMarketImpact**2))
                     *np.sum(np.cosh(ratioAversionOverImpact*t[:-1])*expectedPricePath[:-1]*dt)
                     -(np.sinh(ratioAversionOverImpact*timeHorizon)/(2*coeffMarketImpact**2))
                     *np.sum(np.sinh(ratioAversionOverImpact*t[:-1])*expectedPricePath[:-1]*dt))
        q_0=initialInventory
        if (compute_r0):
            r_0=(-initialPrice/(2*coeffMarketImpact**2)
                 +(ratioAversionOverImpact/np.sinh(ratioAversionOverImpact*timeHorizon))
                 *((liquidationTarget-initialInventory)*np.cosh(ratioAversionOverImpact*timeHorizon)
                   + Kappa_tilde))
        inventory=np.zeros_like(timePartition)
        inventoryRate=np.zeros_like(timePartition)
        inventory[0]=np.array(q_0,copy=True)
        inventoryRate[0]=np.array(r_0,copy=True)
        for i in np.arange(timePartition.shape[0]-1):
            inventoryRate[i+1]=(inventoryRate[i]
                                +(ratioAversionOverImpact**2)*(inventory[i]-liquidationTarget)*dt[i]
                                -priceIncrement[i]/(2*coeffMarketImpact**2))
            inventory[i+1]=(inventory[i]+inventoryRate[i]*dt[i])
        return inventory, inventoryRate
        
    def compute_errorOfLiquidation_and_terminationTime(self,inventory,
                                                       liquidationTarget=0,
                                                       fullOutput=False):
        timePartition=self.pricePath.loc[self.initialTime:self.timeHorizon].index
        errorOfLiquidation=inventory[-1]-liquidationTarget
        isWhen_inventoryBelowTarget=(inventory<=liquidationTarget)
        is_liquidationTerminated=np.any(isWhen_inventoryBelowTarget)
        if (is_liquidationTerminated):
            terminationTime=timePartition[isWhen_inventoryBelowTarget][0]
#             print('compute_errorOfLiquidation_and_terminationTime:')
#             print('    liquidation is terminated at time = {}'.format(terminationTime))
        else:
            terminationTime=self.timeHorizon
        if (fullOutput):
            return is_liquidationTerminated,errorOfLiquidation,terminationTime
        else:
            return errorOfLiquidation
        
    def prolungation_viaEulerLagrange(self,inventory,inventoryRate):
        timePartition=np.linspace(self.initialTime,self.timeHorizon,num=inventory.shape[0])
        rate=inventoryRate[inventoryRate<0][-1]
        additional_time=1.75*np.abs(inventory[-1]/rate)
        terminationTime=self.timeHorizon+additional_time
        price
        inventoryTrajectory_extension,inventoryRate_extension=self.simulate_EulerLagrangeEquation_LTI(
            self.pricePath.loc[self.timeHorizon:terminationTime].values,
            self.expectedPricePath.loc[self.timeHorizon:terminationTime].values,
            initialInventory=inventory[-1],liquidationTarget=self.liquidationTarget,
            coeffMarketImpact=self.coeffMarketImpact,coeffRiskAversion=self.coeffRiskAversion,
            initialTime=self.timeHorizon,timeHorizon=terminationTime,
            compute_r0=False,r_0=rate)
        
        if not (inventoryTrajectory_extension.shape[0]==inventoryRate_extension.shape[0]):
            print('prolungation_viaEulerLagrange: inventoryTrajectory_extension.shape[0]={}'.format(inventoryTrajectory_extension.shape[0]))
            print('        inventoryRate_extension.shape[0]={}'.format(inventoryRate_extension.shape[0]))
        
       
        
        if (np.any(inventoryTrajectory_extension<self.liquidationTarget)):
#             print('prolungation_viaEulerLagrange: np.any(inventoryTrajectory_extension<0)={}'.format(np.any(inventoryTrajectory_extension<self.liquidationTarget)))
#             print('I am truncating at liquidation target')
            index=(inventoryTrajectory_extension>=self.liquidationTarget)
            inventoryTrajectory_extension=inventoryTrajectory_extension[index]
            inventoryRate_extension=inventoryRate_extension[index]
            terminationTime=(self.pricePath.loc[self.timeHorizon:terminationTime].index[index])[-1]
            
            
        if not (np.abs(inventoryTrajectory_extension[-1])<0.005*self.initialInventory):
            print(colored('prolungation_constantRate: WARNING: inventoryTrajectory_extension[-1]={}'
                          .format(inventoryTrajectory_extension[-1]),'red'))
        
                    
        inventory_extended=np.concatenate((inventory,inventoryTrajectory_extension[1:]))
        inventoryRate_extended=np.concatenate(
                    (inventoryRate,
                     rate*np.ones_like(inventoryTrajectory_extension[1:])
                    ))
        
        return inventory_extended, inventoryRate_extended, terminationTime
        
    def prolungation_constantRate(self,
                                  inventory,
                                  inventoryRate):
        timePartition=np.linspace(self.initialTime,self.timeHorizon,num=inventory.shape[0])
        rate=inventoryRate[inventoryRate<0][-1]
        additional_time=np.abs(inventory[-1]/rate)+0.5*np.amin(np.diff(timePartition))
        terminationTime=self.timeHorizon+additional_time
        
        inventoryTrajectory_extension=(inventory[-1]
                                       +rate
                                       *(self.pricePath.loc
                                         [self.timeHorizon:terminationTime].index
                                         -self.timeHorizon))
        if not (np.abs(inventoryTrajectory_extension[-1])<0.0005*self.initialInventory):
            print(colored('prolungation_constantRate: WARNING: inventoryTrajectory_extension[-1]={}'
                          .format(inventoryTrajectory_extension[-1]),'red'))
            print(colored('inventoryTrajectory_extension={}'.format(inventoryTrajectory_extension),'red'))
                    
        inventory_extended=np.concatenate((inventory,inventoryTrajectory_extension[1:]))
        inventoryRate_extended=np.concatenate(
                    (inventoryRate,
                     rate*np.ones_like(inventoryTrajectory_extension[1:])
                    ))
        return inventory_extended, inventoryRate_extended, terminationTime
    
    def prolungation_viaInterpolation(self,inventory,inventoryRate,degree_interpolation=10):
        timePartition=np.linspace(self.initialTime,self.timeHorizon,num=inventory.shape[0])
        epsilon_time=1*np.amin(np.diff(timePartition))
        lastNegativeRate=inventoryRate[inventoryRate<0][-1]
        additional_time=1*np.abs(inventory[-1]/lastNegativeRate)
        extendedTimeHorizon=np.minimum(
            0.975*(self.pricePath.index[-1]),
            1.075*(self.timeHorizon+additional_time))
        terminationTime=np.minimum(1.1*extendedTimeHorizon,self.pricePath.index[-1])
        time_aroundExecutionHorizon=np.linspace(self.timeHorizon,
                                                0.5*(self.timeHorizon
                                                     +np.maximum(0.75*extendedTimeHorizon,1.01*self.timeHorizon)
                                                    )
                                               )
#         print('time_aroundExecutionHorizon[-1]={}'.format(time_aroundExecutionHorizon[-1]))
#         print('extendedTimeHorizon={}'.format(extendedTimeHorizon))
        inventory_aroundExecutionHorizon=(inventory[-1]
                                          +inventoryRate[-1]
                                          *(time_aroundExecutionHorizon
                                            -self.timeHorizon))
        time_beyondTermination=np.linspace(extendedTimeHorizon,2*extendedTimeHorizon)
        x=(np.concatenate(
            (np.concatenate((timePartition,time_aroundExecutionHorizon)),
             time_beyondTermination)))
        y=np.concatenate(
            (np.concatenate((inventory,inventory_aroundExecutionHorizon)),
             np.zeros_like(time_beyondTermination)))
        polynomial_interpol=Polynomial.fit(x,y,
                                           deg=degree_interpolation,
                                           window=[timePartition[0],2*extendedTimeHorizon])
        polynomial_rate=np.polyder(np.poly1d(polynomial_interpol),m=1)
        line_adjustment=Polynomial.fit([self.timeHorizon,self.timeHorizon+2*epsilon_time],
                                       [inventory[-1],polynomial_interpol(self.timeHorizon+2*epsilon_time)],
                                       deg=1)
        line_slope=np.polyder(np.poly1d(line_adjustment))
        inventoryTrajectory_extension=np.concatenate(
            (line_adjustment(self.pricePath.loc[self.timeHorizon:self.timeHorizon+2*epsilon_time].index),
             polynomial_interpol(self.pricePath.loc[self.timeHorizon+2*epsilon_time:terminationTime].index)
            ))
        inventoryRate_extension=np.concatenate(
            (line_slope(self.pricePath.loc[self.timeHorizon:self.timeHorizon+2*epsilon_time].index),
             polynomial_rate(self.pricePath.loc[self.timeHorizon+2*epsilon_time:terminationTime].index)
            ))
        if not (len(inventoryTrajectory_extension)==len(inventoryRate_extension)):
            print(colored('prolungation_viaInterpolation: WARNING: (len(inventoryTrajectory_extension)==len(inventoryRate_extension))={}'
                            .format((len(inventoryTrajectory_extension)==len(inventoryRate_extension))),'red'))
        
        is_inventoryTrajectoryPositive=(inventoryTrajectory_extension>=0)
        inventoryTrajectory_extension=np.array(inventoryTrajectory_extension[is_inventoryTrajectoryPositive],copy=True)
        inventoryRate_extension=np.array(inventoryRate_extension[is_inventoryTrajectoryPositive],copy=True)
        terminationTime=self.pricePath.loc[self.timeHorizon:terminationTime].index[is_inventoryTrajectoryPositive][-1]
        if not (np.abs(inventoryTrajectory_extension[-1]-self.liquidationTarget)<0.005*self.initialInventory):
            print(colored('prolungation_viaInterpolation: WARNING: inventoryTrajectory_extension[-1]={}'
                          .format(inventoryTrajectory_extension[-1]),'red'))
            print(colored('  I am modifying this directly','red'))
            inventoryTrajectory_extension[-1]=self.liquidationTarget
        
        inventory_extended=np.concatenate((inventory,inventoryTrajectory_extension[1:]))
        inventoryRate_extended=np.concatenate((inventoryRate,inventoryRate_extension[1:]))
        return inventory_extended, inventoryRate_extended, terminationTime

        
        
    def handle_termination(self,
                           inventory,
                           inventoryRate,
                           method_extensionOfInventoryTrajectory='constant rate',
                           degree_interpolation=10):
        
        timePartition=np.linspace(self.initialTime,self.timeHorizon,num=inventory.shape[0])
        
        (is_liquidationTerminated,errorOfLiquidation,terminationTime)=self.compute_errorOfLiquidation_and_terminationTime(
            inventory,liquidationTarget=self.liquidationTarget,fullOutput=True)
        
        if (is_liquidationTerminated):
            inventory_interruptedLiquidation=np.array(inventory,copy=True)
            inventory_interruptedLiquidation[timePartition>=terminationTime]=self.liquidationTarget
            inventoryRate_interruptedLiquidation=np.array(inventoryRate,copy=True)
            inventoryRate_interruptedLiquidation[timePartition>=terminationTime]=0.0
            return inventory_interruptedLiquidation,inventoryRate_interruptedLiquidation, terminationTime
        else:
            if (method_extensionOfInventoryTrajectory=='constant rate'):
                inventory_extended, inventoryRate_extended, terminationTime = self.prolungation_constantRate(
                    inventory,inventoryRate)
            elif(method_extensionOfInventoryTrajectory=='interpolation'):
                perc_of_tolerance=2.0
                if(np.abs(inventory[-1]-self.liquidationTarget)<(perc_of_tolerance/100)*self.initialInventory):
                    print('handle_termination: error of liquidation is less than {}% of initial inventory'
                           .format(perc_of_tolerance),
                           '=> despite interpolation was requested, I use constant rate')
                    inventory_extended, inventoryRate_extended, terminationTime = self.prolungation_constantRate(
                        inventory,inventoryRate)
                else:
                    inventory_extended, inventoryRate_extended, terminationTime = self.prolungation_viaInterpolation(
                        inventory,inventoryRate,degree_interpolation=degree_interpolation)
            elif(method_extensionOfInventoryTrajectory=='eulerLagrange'):
                inventory_extended, inventoryRate_extended, terminationTime = self.prolungation_viaEulerLagrange(
                    inventory,inventoryRate)
            return inventory_extended,inventoryRate_extended, terminationTime
    
    def compute_normInventoryTrajectory(self,
                                        inventoryTrajectory,
                                        inventoryRate,
                                        coeffMarketImpact=1,coeffRiskAversion=1):
        if (np.any(inventoryTrajectory==np.nan)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.any(inventoryTrajectory==np.nan)={}'
                          .format(np.any(inventoryTrajectory==np.nan)),'red'))
            print(colored('compute_normInventoryTrajectory: WARNING: inventoryTrajectory=\n {}'
                          .format(inventoryTrajectory),'red'))
        if (np.any(inventoryRate==np.nan)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.any(inventoryRate==np.nan)={}'
                          .format(np.any(inventoryRate==np.nan)),'red'))
            print(colored('compute_normInventoryTrajectory: WARNING: inventoryRate=\n {}'
                          .format(inventoryRate),'red'))
            
        time=inventoryTrajectory.index
        q=np.array((inventoryTrajectory).values,copy=True)[:-1]
        qdot=np.array((inventoryRate).values,copy=True)[:-1]
        q_squared=np.square(q)
        qdot_squared=np.square(qdot)
        if not (np.all(np.diff(time)>=0)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.all(np.diff(time)>=0)={}'
                          .format(np.all(np.diff(time)>=0)),'red'))
        if  (np.any(q==np.nan)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.any(q==np.nan)={}'
                          .format(np.any(q==np.nan)),'red'))
        if  (np.any(qdot==np.nan)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.any(qdot==np.nan)={}'
                          .format(np.any(qdot==np.nan)),'red'))
        if not (np.all(q_squared>=0)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.all(q_squared>=0)={}'
                          .format(np.all(q_squared>=0)),'red'))
        if not (np.all(qdot_squared>=0)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.all(qdot_squared>=0)={}'
                          .format(np.all(qdot_squared>=0)),'red'))
        if not (np.all(time==inventoryRate.index)):
            print(colored('compute_normInventoryTrajectory: WARNING: np.all(time==inventoryRate.index)={}'
                         .format(np.all(time==inventoryRate.index)),'red'))
        
        pathwiseNorm=np.sqrt(
            (coeffMarketImpact**2)
            *np.sum(
                qdot_squared
                *np.diff(time))
            +(coeffRiskAversion**2)
            *np.sum(
                q_squared
                *np.diff(time))
        )
        if not (pathwiseNorm>=0):
            print(colored('compute_normInventoryTrajectory: WARNING: pathwiseNorm={}'
                          .format(pathwiseNorm),'red'))
        return pathwiseNorm
    
    def compute_xi(self,
                   pricePath,expectedPricePath,
                   coeffMarketImpact=1.0,
                   coeffRiskAversion=1.0,
                   timeHorizon=1.0):
        pricePath=pd.Series(pricePath[pricePath.index<timeHorizon],copy=True)
        expectedPricePath=pd.Series(expectedPricePath[expectedPricePath.index<timeHorizon],copy=True)
        time=np.concatenate((pricePath.index,[timeHorizon]))
        t=np.array(time[:-1],copy=True)
        if not (np.all(t==expectedPricePath.index)):
            print(colored('compute_xi: WARNING: t=expectedPricePath.index)={}'
                         .format(np.all(t==expectedPricePath.index)),'red'))
        
        
        ratioAversionOverImpact=coeffRiskAversion/coeffMarketImpact
        c_1=coeffMarketImpact
        c_2=coeffRiskAversion
        c_3=ratioAversionOverImpact
        T=np.array(timeHorizon,copy=True)
        first_summand=(2*c_1*c_2
                       *(self.liquidationTarget-self.initialInventory)/(np.sinh(c_3*T)))
        second_summand=(-c_3*
                        np.sum(
                            np.sinh(c_3*(T-t))
                            *pricePath
                            *np.diff(time)
                        ))
        third_summand=((c_3*np.cosh(c_3*T)/np.sinh(c_3*T))
                       *np.sum(
                            np.cosh(c_3*(T-t))
                            *expectedPricePath
                            *np.diff(time)
                       ))
        xi=1/(np.abs(first_summand+second_summand+third_summand))
        return xi
    
    def compute_riskAdjustedCostOfExecution(self,inventory,inventoryRate):
        c_1=self.coeffMarketImpact
        c_2=self.coeffRiskAversion
        time=np.array(inventory.index,copy=True)
        if not (np.all(time==inventoryRate.index)):
            print(colored('compute_riskAdjustedCostOfExecution: WARNING: np.all(time==inventoryRate.index)={}'
                          .format(np.all(time==inventoryRate.index)),'red'))
        price=pd.Series(self.pricePath.loc[:time[-1]],copy=True)
        if not (np.all(time==price.index)):
            print(colored('compute_riskAdjustedCostOfExecution: WARNING: np.all(time==price.index)={}'
                          .format(np.all(time==price.index)),'red'))
            print(colored('     time.shape={},  price.shape={}'
                          .format(time.shape,price.shape),'red'))
            print(colored('     time={},  price.index={}'
                          .format(time,price.index),'red'))
        
        first_summand=inventoryRate*price
        second_summand=np.square(c_1*inventoryRate)
        third_summand=np.square(c_2*(inventory-self.liquidationTarget))
        integrand=np.array((first_summand+second_summand+third_summand).values,copy=True)
        if not (integrand.shape==time.shape):
            print(colored('compute_riskAdjustedCostOfExecution: WARNING: (integrand.shape==time.shape)={}'
                          .format((integrand.shape==time.shape)),'red'))
            print(colored('     integrand.shape={},  time.shape={}'
                          .format(integrand.shape,time.shape),'red'))
        integral=np.sum(integrand[:-1]*np.diff(time))
        return integral
    
    def compute_revenuesFromTrade(self,inventoryRate):
        c_1=self.coeffMarketImpact
        c_2=self.coeffRiskAversion
        time=np.array(inventoryRate.index,copy=True)
        price=pd.Series(self.pricePath.loc[:time[-1]],copy=True)
        if not (np.all(time==price.index)):
            print(colored('compute_riskAdjustedCostOfExecution: WARNING: np.all(time==price.index)={}'
                          .format(np.all(time==price.index)),'red'))
            print(colored('     time.shape={},  price.shape={}'
                          .format(time.shape,price.shape),'red'))
            print(colored('     time={},  price.index={}'
                          .format(time,price.index),'red'))
        
        first_summand=-inventoryRate*price
        second_summand=-np.square(c_1*inventoryRate)
        integrand=np.array((first_summand+second_summand).values,copy=True)
        if not (integrand.shape==time.shape):
            print(colored('compute_riskAdjustedCostOfExecution: WARNING: (integrand.shape==time.shape)={}'
                          .format((integrand.shape==time.shape)),'red'))
            print(colored('     integrand.shape={},  time.shape={}'
                          .format(integrand.shape,time.shape),'red'))
        integral=np.sum(integrand[:-1]*np.diff(time))
        return integral
        
        
        
        
                
        
    
        


# In[ ]:


class price(object):
    def __init__(self,timeWindow,
                 initialPrice=100.0,
                 terminalValue=100.0,
                 volatility=0.2,
                 hurst_exponent=0.5,
                 stochastic_process='arithmetic BM',
                 num_of_simulations=100,
                 plot_price=False):
        self.initialPrice=initialPrice
        self.terminalValue=terminalValue
        self.volatility=volatility
        self.timeWindow=timeWindow
        samples,expectedPricePath=self.simulate_pricePath(self.timeWindow,
                                                          initialPosition=self.initialPrice,
                                                          terminalValue=self.terminalValue,
                                                          volatility=self.volatility,
                                                          hurst_exponent=hurst_exponent,
                                                          numOfSamples=num_of_simulations,
                                                          stochastic_process=stochastic_process)
        self.expectedPricePath=pd.Series(expectedPricePath,index=timeWindow)
        self.samples=pd.DataFrame(samples,index=timeWindow)
        highest_point=np.amax(samples,axis=0)
        lowest_point=np.amin(samples,axis=0)
        index_high=np.argmax(highest_point)
        index_low=np.argmin(lowest_point)

#             print('samples.shape={}'.format(samples.shape))
#             print('quadratic_variation.shape={}'.format(quadratic_variation.shape))
#             print('highest_point.shape={}'.format(highest_point.shape))
#             print('lowest_point.shape={}'.format(lowest_point.shape))
#             print('increments.shape={}'.format(increments.shape))
            
#         pricePath_vol=np.array(samples[:,np.argmax(quadratic_variation)],copy=True)
        pricePath_high=np.array(samples[:,index_high],copy=True)
        pricePath_low=np.array(samples[:,index_low],copy=True)
#         pricePath_vol=np.array(samples[:,np.random.randint(samples.shape[1])],copy=True)
        pricePath_2=np.array(samples[:,np.random.randint(samples.shape[1])],copy=True)
        pricePath_3=np.array(samples[:,np.random.randint(samples.shape[1])],copy=True)
                    
        
        
#         self.pricePath_vol=pd.Series(pricePath_vol,index=timeWindow)
        self.pricePath_high=pd.Series(pricePath_high,index=timeWindow)
        self.pricePath_low=pd.Series(pricePath_low,index=timeWindow)
        self.pricePath_2=pd.Series(pricePath_2,index=timeWindow)
        self.pricePath_3=pd.Series(pricePath_3,index=timeWindow)
        self.pricePaths=pd.DataFrame({'expected price':self.expectedPricePath,
                                      'price 1':self.pricePath_high,
                                      'price 2':self.pricePath_2,
                                      'price 3':self.pricePath_3,
                                      'price 4':self.pricePath_low
                                      })
        if (plot_price):
            fig = plt.figure(figsize=(20, 8))
            cmap = plt.get_cmap('viridis')
            ax_price= fig.add_subplot(111)
            self.pricePaths.plot(ax=ax_price,cmap=cmap)
            ax_price.set_xlabel('time')
            ax_price.set_ylabel('price')
            ax_price.set_title('price process as {}'.format(stochastic_process))
            plt.show()
    
    def simulate_pricePath(self,timeWindow,initialPosition=100.0,
                           terminalValue=100.0,
                           volatility=0.1,
                           numOfSamples=1,
                           hurst_exponent=0.5,
                           stochastic_process='arithmetic BM'):
        if (np.logical_or(
            (np.logical_or((stochastic_process=='arithmetic BM')
                           ,(stochastic_process=='aBM'))),
            (stochastic_process=='brownian motion'))):
            stochastic_process='aBM'
        elif(np.logical_or(
            (np.logical_or((stochastic_process=='geometric BM')
                           ,(stochastic_process=='gBM'))),
            (stochastic_process=='geom brownian motion'))):
            stochastic_process='gBM'
        elif(np.logical_or(
            (np.logical_or(
                (np.logical_or((stochastic_process=='brownianBridge'),
                               (stochastic_process=='bridge'))),
                (stochastic_process=='Brownian Bridge'))),
            stochastic_process=='brownian bridge')):
            stochastic_process='brownianBridge'
        elif(np.logical_or(
            (np.logical_or((stochastic_process=='exponentialBridge')
                           ,(stochastic_process=='exp bridge'))),
            (stochastic_process=='exponential brownian bridge'))):
            stochastic_process='exponentialBridge'
        elif(np.logical_or(
            (np.logical_or((stochastic_process=='fractional BM')
                           ,(stochastic_process=='fBM'))),
            (stochastic_process=='fractional brownian motion'))):
            stochastic_process='fBM'
        
        timeHorizon=timeWindow[-1]
        t=np.repeat(np.expand_dims(timeWindow,axis=1),numOfSamples,axis=1)
        timeDiff=np.repeat(np.expand_dims(np.sqrt(np.diff(timeWindow)),axis=1),numOfSamples,axis=1)
        noise=np.concatenate((np.zeros((1,numOfSamples)),
                              (normalDistribution.rvs(size=(timeWindow.shape[0]-1,numOfSamples))
                               *timeDiff)
                             ),axis=0)
        standard_BM=np.cumsum(noise,axis=0)
        if (stochastic_process=='aBM'):
            pricePath=initialPosition+volatility*standard_BM
            expectedPricePath=initialPosition* np.ones(pricePath.shape[0])
        elif(stochastic_process=='gBM'):
            pricePath=initialPosition*np.exp(
                volatility*standard_BM
                -0.5*np.square(volatility)*t)
            expectedPricePath=initialPosition* np.ones(pricePath.shape[0])
        elif(stochastic_process=='brownianBridge'):
            brownianMotion=initialPosition+volatility*standard_BM
            terminalBrownianPosition=np.repeat(np.expand_dims(brownianMotion[-1,:],axis=0),brownianMotion.shape[0],axis=0)
#             print('t.shape={}'.format(t.shape))
#             print('brownianMotion.shape={}'.format(brownianMotion.shape))
#             print('terminalBrownianPosition.shape={}'.format(terminalBrownianPosition.shape))  
            pricePath=brownianMotion+(t/timeHorizon)*(terminalValue-terminalBrownianPosition)
            expectedPricePath=(1-timeWindow/timeHorizon)*initialPosition+(timeWindow/timeHorizon)*terminalValue
        elif(stochastic_process=='exponentialBridge'):
            brownianMotion=np.log(initialPosition)+volatility*standard_BM
            terminalBrownianPosition=np.repeat(np.expand_dims(brownianMotion[-1,:],axis=0),brownianMotion.shape[0],axis=0)
            pricePath=np.exp(brownianMotion+(t/timeHorizon)*(np.log(terminalValue)-terminalBrownianPosition))
            expectedPricePath=np.exp((timeWindow/timeHorizon)*np.log(terminalValue)
                                     +0.5*np.square(volatility)*(timeWindow-np.square(timeWindow)/timeHorizon)
                                     +(1-timeWindow/timeHorizon)*np.log(initialPrice))
        elif(stochastic_process=='fBM'):
            print('sampling fBM, with H={}'.format(hurst_exponent))
            num_partitionPoints=timeWindow.shape[0]-1
            samples=np.zeros((timeWindow.shape[0],numOfSamples))
            fractional_BM=FBM(n=num_partitionPoints, 
                              hurst=hurst_exponent,
                              length=(timeWindow[-1]-timeWindow[0]), 
                              method='daviesharte')
            for k in np.arange(numOfSamples):
                samples[:,k]=fractional_BM.fbm()
            pricePath=initialPosition+volatility*samples
            expectedPricePath=initialPosition* np.ones(pricePath.shape[0])
    
            
            
        
        return pricePath,expectedPricePath
        


        
def produce_labelsForInventories(n=1):
    label=('static inventory')
    for k in np.arange(1,n+1):
        label=np.append(label,
                       'inventory {}'.format(k))
    return label