#!/bin/usr/python env
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

def plot_price_and_rate(liquidator, num=2):#accept an instance of the class Liquidator of liquidation.pyx
    liq=liquidator
    T = np.amax(liq.simulated_liquidation.simulations['termination_times'])
    order=list(liq.simulated_liquidation.simulations['df_termin']['plot_order'].values)
    num=min(num,len(order))
    order=order[:num]
    fig=plt.figure(figsize=(15,8))
    cmap = plt.get_cmap('winter')
    axp=fig.add_subplot(111)
    dfp=liq.simulated_liquidation.df_prices
    dfp=dfp.loc[dfp['time']<=T]
    cols=['time']
    for k in order:
        cols.append('price_{}'.format(k))
    dfp=dfp.loc[:,cols].copy()
    dfp.set_index('time', inplace=True)
    dfp.plot(ax=axp,colormap=cmap)
    exprice=liq.priceprocess.expected_trajectory
    exprice=exprice[exprice[:,0]<=T]
    axp.plot(exprice[:,0],exprice[:,1],label='reversion_target',color='lightblue',linestyle='--',linewidth=3)
    axi=axp.twinx()
    cols=['time']
    df=liq.simulated_liquidation.df_simul
    for k in order:
        cols.append('rate_{}'.format(k))
    df=df.loc[:,cols].copy()
    df.set_index('time', inplace=True)
    df=-df
    df.plot(ax=axi,colormap=cmap)

    ylim=[0.95*dfp.iloc[:,1].min(),1.001*dfp.iloc[:,1].max()]
    axp.set_ylim(ylim)
    axi.legend(loc=3)
    axp.legend(loc=6)
    axi.set_ylabel('absolute rate')
    axp.set_ylabel('price')
    axp.set_xlabel('time')
    plt.show()
def plot_price_and_inventory(liquidator, num=2): #accept an instance of the class Liquidator of liquidation.pyx
    liq=liquidator
    T = np.amax(liq.simulated_liquidation.simulations['termination_times'])
    order=list(liq.simulated_liquidation.simulations['df_termin']['plot_order'].values)
    num=min(num,len(order))
    order=order[:num]
    fig=plt.figure(figsize=(15,8))
    cmap = plt.get_cmap('winter')
    axp=fig.add_subplot(111)
    dfp=liq.simulated_liquidation.df_prices
    dfp=dfp.loc[dfp['time']<=T]
    cols=['time']
    for k in order:
        cols.append('price_{}'.format(k))
    dfp=dfp.loc[:,cols].copy()
    dfp.set_index('time', inplace=True)
    dfp.plot(ax=axp,colormap=cmap)
    exprice=liq.priceprocess.expected_trajectory
    exprice=exprice[exprice[:,0]<=T]
    axp.plot(exprice[:,0],exprice[:,1],label='reversion_target',color='lightblue',linestyle='--',linewidth=3)
    axi=axp.twinx()
    cols=['time']
    df=liq.simulated_liquidation.df_simul
    for k in order:
        cols.append('inventory_{}'.format(k))
    df=df.loc[:,cols].copy()
    df.set_index('time', inplace=True)
    df.plot(ax=axi,colormap=cmap)

    ylim=[0.95*dfp.iloc[:,1].min(),1.0*dfp.iloc[:,1].max()]
    axp.set_ylim(ylim)
    axi.legend(loc=3)
    axp.legend(loc=6)
    plt.show()




#class SimulatedLiquidation:
#    def __init__(self,simulations=None, prices=None):
#        if simulations!=None and prices!=None:
#            self.get_data(simulations, prices)
#    def get_data(self, simulations, prices):
#        print("storing data")
#        self.get_prices(prices)
#        self.simulations=copy.copy(simulations)
#        dfs=[]
#        for k in range(len(simulations['trajectories'])):
#            cols=['time', 'inventory_{}'.format(k), 'rate_{}'.format(k)]
#            dfs.append(pd.DataFrame(simulations['trajectories'][k],columns=cols))
#            if k==0:
#                df=dfs[0].copy()
#            else:
#                df=df.merge(dfs[k],how='outer',on='time')
#        self.simulations['dfs']=dfs
#        self.df_simul = df
#        self.store_terminationtimes()
#    def get_prices(self, prices):
#        print("storing prices")
#        self.prices=copy.copy(prices)
#        dfs=[]
#        for k in range(len(prices)):
#            cols=['time', 'price_{}'.format(k)]
#            dfs.append(pd.DataFrame(prices[k],columns=cols))
#            if k==0:
#                df=dfs[0].copy()
#            else:
#                df=df.merge(dfs[k],how='outer',on='time')
#        self.df_prices=df
#    def store_terminationtimes(self,):
#        print("storing termination times")
#        tt=self.simulations['termination_times']
#        df=pd.DataFrame({'simulation_id': np.arange(len(tt)),'termination':tt})
#        ott=np.array(df.sort_values(by='termination')['simulation_id'].values)
#        lstt=np.zeros(len(df), dtype=np.int)
#        l=len(ott)
#        lstt[1::2]=ott[:l-l//2-1:-1]
#        lstt[::2]=ott[:l//2+l%2]
#        df['plot_order']=lstt
#        self.simulations['df_termin']=df




