import sys
import os
cdef str path_cwd = os.getcwd()
import numpy as np
cimport numpy as np
import pandas as pd
import bisect
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

class Price:
    def __init__(self,lobster_midprice, keep_lobster_dollarunit=False):
        self.get_lobster_midprice(lobster_midprice,keep_lobster_dollarunit)
    def get_lobster_midprice(self,midprice,keep_lobster_dollarunit=False):
        self.lobster_midprice=midprice
        self.store_data(keep_lobster_dollarunit)
        self.store_logprice()
    def store_data(self, keep_lobster_dollarunit=False):
        cdef np.ndarray[DTYPEf_t, ndim=2] data = np.array(self.lobster_midprice.values,dtype=DTYPEf)
        #normalise and center times
        data[:,0]-= data[0,0]
        data[:,0]/=np.amax(data[:,0])
        if not keep_lobster_dollarunit:
            if np.mean(data[:,1])>1.0e+04:
                data[:,1]*=1.0e-04
        self.data=data
    def store_logprice(self):
        cdef np.ndarray[DTYPEf_t, ndim=2] logprice = np.array(self.data, copy=True)
        logprice[:,1]=np.log(logprice[:,1])
        self.logprice=logprice
    def set_forecast(self,np.ndarray[DTYPEf_t, ndim=2] forecast): #tipical call: self.set_forecast(self.fourier.series_val)
        self.forecast=forecast #expected in log-scale
        cdef np.ndarray[DTYPEf_t, ndim=2] expected_trajectory = np.array(forecast, copy=True)
        expected_trajectory[:,1]=np.exp(forecast[:,1])
        self.expected_trajectory=expected_trajectory
    def set_fourier(self, int truncation = 0):
        fourier=FourierSeries(truncation = truncation)
        fourier.get_data(self.logprice)
        fourier.store_fourier_coef()
        self.fourier=fourier
    def set_ou(self):
        ou=OrnsteinUhlenbeck()
        cdef np.ndarray[DTYPEf_t, ndim=2] oudata = np.array(self.fourier.data, copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=2] fseries = self.fourier.series_val
        cdef int n, idx
        for n in range(oudata.shape[0]):
            idx=bisect.bisect_left(fseries[:,0], oudata[n,0])
            oudata[n,1]-=fseries[idx,1]
        ou.get_data(oudata)
        self.ou=ou
    def calibrate_ou(self, int maxiter=10, int num_batches=10, DTYPEf_t caliber = 10.0):
        ou=self.ou
        for n in range(1+maxiter):
            ou.estimate_volatility(num_batches)
            ou.estim_rate()
            ou.mle_estim()
            ou.set_sigma(ou.vol)
            ou.detect_jumps(caliber)
            if n==0:
                self.jumpsizes=np.array(ou.jumpsizes,copy=True)
                self.jumptimes=np.array(ou.jumptimes,copy=True)
            idxnojumps=np.logical_not(ou.idxjumps)
            data=ou.data[idxnojumps,:]
            ou.get_data(data)
        self.ou = ou
    def set_pointprocess(self,):
        cdef np.ndarray[DTYPEf_t, ndim=2] data =\
                np.concatenate(
                        [np.expand_dims(self.jumptimes,axis=1),
                            np.expand_dims(self.jumpsizes,axis=1)],
                        axis=1)
        pointprocess=PointProcess()
        pointprocess.get_data(data)
        self.pointprocess=pointprocess
    def calibrate_pointprocess(self,int maxiter=100):
        self.pointprocess.estimate_marksratio()
        self.pointprocess.estimate_signedmarks(sign=1)
        self.pointprocess.estimate_signedmarks(sign=-1)
        self.pointprocess.estimate_poisson()
    def simulate(self,DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0, Py_ssize_t maxnum_events=10**7, int howmany=1):
        assert howmany>0
        if t1<=t0:
            t0=self.data[0,0]
            t1=self.data[len(self.data)-1,0]
        t1=min(t1,self.forecast[len(self.forecast)-2,0])
        idx_timewindow = np.logical_and(self.data[:,0]>=t0, self.data[:,0]<=t1)
        cdef np.ndarray[DTYPEf_t, ndim=2] times = t0*np.ones((1+maxnum_events, howmany),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=2] logprice = np.zeros((1+maxnum_events, howmany), dtype=DTYPEf)
        cdef int idxforecast = bisect.bisect(self.forecast[:,0], t0)
        cdef np.ndarray[DTYPEf_t, ndim=2] trendcorr = self.forecast[idxforecast,1]*np.ones((1+maxnum_events,howmany),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = np.diff(self.data[idx_timewindow,0])
        cdef DTYPEf_t T=t0
        cdef list logsimulations=[], simulations=[]
        cdef int n=0, m=0, k=0
        for k in range(howmany):
            print("simulation number: {}".format(1+k))
            idxforecast = bisect.bisect(self.forecast[:,0], t0)
            n, m = 0, 0
            while times[n,k]<t1 and n<maxnum_events:
                nextpnt=self.pointprocess.sample_nextpnt()
                T=times[n,k]+nextpnt['dt']
                while (times[n,k]+dt[m]) < T:
                    times[n+1,k]=times[n,k]+dt[m]
                    logprice[n+1,k]=self.ou.sample_increment(initpos=logprice[n,k], dt=dt[m])
                    while times[n+1,k]>self.forecast[idxforecast,0]:
                        idxforecast+=1
                    trendcorr[n+1,k]=self.forecast[idxforecast,1]
                    n+=1
                    m+=1
    #            print('jumptime: times[n]={}, times[n]+dt[m]={}, T={}'.format(times[n], times[n]+dt[m], T))
                times[n+1,k]=T
                while times[n+1,k]>self.forecast[idxforecast,0]:
                    idxforecast+=1
                logprice[n+1,k]=logprice[n,k]+nextpnt['mark']
                trendcorr[n+1,k]=self.forecast[idxforecast,1]  
                n+=1
            logprice[:,k]+=trendcorr[:,k]
            logsimul = \
                np.concatenate([
                    np.expand_dims(times[:n,k],axis=1),
                    np.expand_dims(logprice[:n,k], axis=1)
                    ],axis=1)
            simul = np.array(logsimul, copy=True)
            simul[:,1] = np.exp(logsimul[:,1])
            logsimulations.append(np.array(logsimul, copy=True))
            simulations.append(np.array(simul, copy=True))
        self.simulations=simulations
        self.logsimulations=logsimulations
        cdef np.ndarray[DTYPEf_t, ndim=2] log_sim = np.array(logsimulations[0],copy=True)
        cdef np.ndarray[DTYPEf_t, ndim=2] sim = np.array(simulations[0],copy=True)
        self.simulation=sim
        self.logsimulation=log_sim
    def plot_data(self,DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0, logscale=False, show_forecast=False):
        if t1<=t0:
            t0=self.data[0,0]
            t1=self.data[len(self.data)-1,0]
        idx_timewindow=np.logical_and(self.data[:,0]>=t0, self.data[:,0]<=t1)
        cdef np.ndarray[DTYPEf_t, ndim=1] times = self.data[idx_timewindow,0]
        cdef np.ndarray[DTYPEf_t, ndim=1] data = self.data[idx_timewindow,1]
        if show_forecast:
            idx=np.logical_and(self.forecast[:,0]>=t0, self.forecast[:,0]<=t1)
            forecast=self.forecast[idx,:]
        if logscale:
            data=np.log(data)
            ylabel='log-price'
        else:
            ylabel='price'
            if show_forecast:
                forecast[:,1]=np.exp(forecast[:,1])
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        ax.plot(times, data, label='data', color='blue')
        if show_forecast:
            ax.plot(forecast[:,0], forecast[:,1], label='reversion_target', color='lightblue', linestyle='--', linewidth=3)
        ax.set_xlabel('time')
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()
    def plot(self,DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0, logscale=False, show_data=False):
        forecast=np.array(self.forecast,copy=True)
        if logscale:
            price=np.array(self.logsimulation,copy=True)
            ylabel='log-price'
            if show_data:
                data=np.array(self.logprice, copy=True)
        else:
            price=np.array(self.simulation, copy=True)
            forecast[:,1]=np.exp(forecast[:,1])
            ylabel='price'
            if show_data:
                data=np.array(self.data,copy=True)
        if t1<=t0:
            t0=self.data[0,0]
            t1=self.data[len(self.data)-1,0]
        idx_timewindow=np.logical_and(price[:,0]>=t0, price[:,0]<=t1)
        price=price[idx_timewindow,:]
        idx_=np.logical_and(forecast[:,0]>=t0, forecast[:,0]<=t1)
        forecast=forecast[idx_,:]
        if show_data:
            idx_=np.logical_and(data[:,0]>=t0,data[:,0]<=t1)
            data=data[idx_,:]
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        if show_data:
            ax.plot(data[:,0], data[:,1], color='blue', label='data')
        ax.plot(price[:,0], price[:,1], color='green', label='simulation')
        ax.plot(forecast[:,0], forecast[:,1], color='lightblue', linestyle='--', linewidth=3, label='reversion_target')
        ax.set_xlabel('time')
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.show()

cdef struct Stats:
    DTYPEf_t mean
    DTYPEf_t std
    DTYPEf_t absmin
    DTYPEf_t absmax
    int numlevels
cdef struct Marks:
    DTYPEf_t ratio
    Stats pos
    Stats neg
cdef struct Poisson:
    DTYPEf_t rate
    DTYPEf_t scale
cdef struct NextPoint:
    DTYPEf_t dt
    DTYPEf_t mark
class PointProcess:
    def __init__(self,):
        cdef Poisson poisson
        self.poisson=poisson
        cdef Marks marks
        self.marks = marks
    def get_data(self, np.ndarray[DTYPEf_t, ndim=2] data):
        print('PointProcess.get_data: shape=({},{})'.format(data.shape[0],data.shape[1]))
        self.data=data
    def estimate_marksratio(self,):
        cdef int numpos= np.sum(self.data[:,1]>0)
        cdef int numneg=np.sum(self.data[:,1]<0)
        cdef DTYPEf_t ratio = float(numpos)/float(numpos+numneg)
        self.marks['ratio'] = ratio
    def estimate_signedmarks(self,int sign = 1,):
        if (sign!=1) and (sign!=-1):
            print("\n Error: 'sign' must be either +1 or -1, but {} was passed".format(sign))
            raise ValueError()
        idxsign=(sign*self.data[:,1])>0
#        vals is the array of absolute marks of the considered direction (eiter sign=+1 or sign =-1)
        cdef np.ndarray[DTYPEf_t, ndim=1] vals = np.array(sign*self.data[idxsign,1])
        cdef DTYPEf_t mean = sign*np.mean(vals)
        cdef DTYPEf_t std = np.std(vals)
        cdef DTYPEf_t absmin = np.amin(vals)
        cdef DTYPEf_t absmax = np.amax(vals)
#        For the following few lines, we exponentiate the price scale, due to the original consideration of log prices 
        vals=np.exp(vals)
        cdef np.ndarray[DTYPEf_t, ndim=1] deltas = np.abs(np.diff(vals))
        idxdelta=(deltas>0.0)
        cdef DTYPEf_t delta = np.amin(deltas[idxdelta])
        assert delta>0
        cdef int levels = 1+int(np.exp(absmax-absmin)//delta)
        assert levels>0
        cdef np.ndarray[DTYPEf_t, ndim=2] distrib = np.zeros((levels,2),dtype=DTYPEf)
        cdef DTYPEf_t startval = exp(absmin) #starting point of the grid of prices
        distrib[:,0]=startval+delta*np.arange(levels,dtype=DTYPEf) #grid of prices
        cdef DTYPEf_t low=0.0, high=1.0
        cdef int n, j
        for n in range(len(vals)):
            for j in range(levels):
                low=startval+j*delta
                high=startval+(j+1)*delta
                if (vals[n]>=low) and (vals[n]<high):
                    distrib[j,1]+=1
                    break
        idx=distrib[:,1]>0.0
        levels=np.sum(idx)
        cdef np.ndarray[DTYPEf_t, ndim=2] probs = np.array(distrib[idx,:],copy=True)
        probs[:,0]=sign*np.log(probs[:,0]) #we go back to the original logscale and we incorporate the sign of marks
        probs[:,1]/=np.sum(probs[:,1]) #normalisation to turn frequencies into probabilities
        if sign == 1:
            direct='pos'
            self.pos_distrib=probs
        else:
            direct='neg'
            self.neg_distrib=probs
        self.marks[direct]['mean']=mean
        self.marks[direct]['std']=std
        self.marks[direct]['absmin']=absmin
        self.marks[direct]['absmax']=absmax
        self.marks[direct]['numlevels']=levels
    def estimate_poisson(self,):
        cdef DTYPEf_t avg = np.mean(np.diff(self.data[:,0]))
        assert avg>0.0
        cdef DTYPEf_t rate = 1.0/avg
        cdef DTYPEf_t scale = 1.0/rate
        self.poisson['rate'] = rate
        self.poisson['scale'] = scale
    def sample_nextpnt(self,):
        cdef DTYPEf_t dt = np.random.exponential(1/self.poisson['rate'])
        if (float(rand())/float(RAND_MAX)) < self.marks['ratio']:
            arr=self.pos_distrib[:,0]
            probs=self.pos_distrib[:,1]
        else:
            arr=self.neg_distrib[:,0]
            probs=self.neg_distrib[:,1]
        cdef DTYPEf_t mark = np.random.choice(arr,p=probs)
        cdef NextPoint res
        res.dt=dt
        res.mark=mark
        return res
    def simulate(self,DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0, int max_number_of_events = 100000):
        if t1<t0:
            t0=self.data[0,0]
            t1=self.data[len(self.data)-1,0]
        cdef np.ndarray[DTYPEf_t, ndim=1] times = np.zeros((1+max_number_of_events,),dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] sizes = np.zeros((1+max_number_of_events,),dtype=DTYPEf)
        times[0]=t0
        cdef int n=0
        cdef DTYPEf_t scale=1/self.poisson['rate']
        cdef DTYPEf_t chance=0.0, RANDMAX = float(RAND_MAX)
        while ((times[n]<t1) and (n<max_number_of_events)):
            times[n+1]=times[n]+np.random.exponential(scale)
            chance = float(rand())/RANDMAX
            if chance<self.marks['ratio']:
                arr=self.pos_distrib[:,0]
                probs=self.pos_distrib[:,1]
            else:
                arr=self.neg_distrib[:,0]
                probs=self.neg_distrib[:,1]
            sizes[n+1]=np.random.choice(arr,p=probs)
            n+=1
        cdef np.ndarray[DTYPEf_t, ndim=1] jumptimes = times[1:n+1]
        cdef np.ndarray[DTYPEf_t, ndim=1] jumpsizes = sizes[1:n+1]
        cdef np.ndarray[DTYPEf_t, ndim=2] simulation =\
                np.concatenate(
                        [np.expand_dims(jumptimes,axis=1),
                            np.expand_dims(jumpsizes,axis=1)],
                        axis=1)
        self.simulation=simulation
    def plot(self, DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0, only_data=False):
        if t1<t0:
            t0=self.data[0,0]
            t1=self.data[len(self.data)-1,0]
        idxdata=np.logical_and(self.data[:,0]>=t0, self.data[:,0]<=t1)
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        size=5.0
        ax.scatter(self.data[idxdata,0],self.data[idxdata,1],s=size,color='blue',label='data')
        if not only_data:
            idxsim=np.logical_and(self.simulation[:,0]>=t0, self.simulation[:,0]<=t1)
            ax.scatter(self.simulation[idxsim,0],self.simulation[idxsim,1],s=size,color='green',alpha=0.2,label='simulation')
        ax.set_xlabel('time')
        ax.set_ylabel('log-price')
        ax.legend()
        plt.show()

class OrnsteinUhlenbeck:
    def __init__(self,):
        pass
    def get_data(self,np.ndarray[DTYPEf_t, ndim=2] data):
        print('OrnsteinUhlenbeck.get_data: shape=({},{})'.format(data.shape[0],data.shape[1]))
        if not np.isclose(data[0,0],0.0):
            print("WARNING: OrsteinUhlenbeck.get_data: time does not start from 0")
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = np.diff(data[:,0])
        self.dt=dt
        self.data=data
    def set_rate(self,DTYPEf_t rate):
        self.rate=rate
    def set_longterm_mean(self, DTYPEf_t mu):
        self.longterm_mean=mu
    def set_sigma(self, DTYPEf_t sigma):
        self.sigma = sigma
    def estimate_volatility(self,int num_batches=1):
        data=self.data
        cdef int len_ = num_batches*(num_batches+1)//2
        cdef np.ndarray[DTYPEf_t, ndim=1] qv = np.zeros((len_,), dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=2] interval = np.zeros((len_,1), dtype=DTYPEf)
        cdef int n, i, l, r, batchsize, startp
        for n in range(1,1+num_batches):
            batchsize=len(data)//n
            for i in range(n):
                l, r, startp = batchsize*(i), batchsize*(i+1), n*(n-1)//2
                qv[startp+i]=np.sum(np.square(np.diff(self.data[l:r,1])))
                interval[startp+i,0]=self.data[min(len(data)-1,r),0] - self.data[l,0]
        cdef DTYPEf_t vol2 = np.squeeze(np.linalg.lstsq(interval, qv, rcond=None)[0])
        cdef DTYPEf_t vol = np.sqrt(max(0.0,vol2))
        print("vol={}".format(vol))
        self.vol = vol
    def estim_rate(self,):
        cdef DTYPEf_t var = np.var(self.data[:,1])
        cdef DTYPEf_t rate = self.vol**2/(2*var)
        print("convergence_rate={}".format(rate))
        self.convergence_rate = rate 
    def mle_estim(self,):
        rate=self.convergence_rate
        cdef np.ndarray[DTYPEf_t, ndim=1] x = self.data[:,1]
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = self.dt
        cdef np.ndarray[DTYPEf_t, ndim=1] exp_adt = np.exp(-rate*dt)
        cdef DTYPEf_t num=np.sum(np.divide(x[1:]-exp_adt*x[:len(x)-1],1+exp_adt))
        cdef DTYPEf_t den=np.sum(np.divide(1-exp_adt,1+exp_adt))
        cdef DTYPEf_t mu = num/den
        self.longterm_mean = mu
        print("longterm_mean={}".format(mu))
        cdef DTYPEf_t sigma2 = (2*rate/len(x))\
                *np.sum(np.divide(
                        np.square(x[1:]-mu - exp_adt*(x[:len(x)-1]-mu)),
                        1-np.square(exp_adt)
                        ))
        cdef DTYPEf_t sigma = np.sqrt(sigma2)
        self.sigma = sigma
        print("mle_sigma={}".format(sigma))
    def detect_jumps(self, DTYPEf_t caliber = 3.0):
        rate=self.convergence_rate
        cdef np.ndarray[DTYPEf_t, ndim=1] std=self.sigma*np.sqrt((1.0-np.exp(-2*rate*self.dt))/(2*rate))
        cdef np.ndarray[DTYPEf_t, ndim=1] dx = np.diff(self.data[:,1])
        idx = (np.abs(dx) >= caliber*std)
        cdef np.ndarray[DTYPEf_t, ndim=1] jumpsizes = np.array(dx[idx], dtype=DTYPEf)
        idx = np.array(np.concatenate([np.zeros((1,),dtype=np.bool),idx], axis=0),dtype=np.bool)
        print("{} jumps detected". format(np.sum(idx)))
        cdef np.ndarray[DTYPEf_t, ndim=1] jumptimes = self.data[idx,0]
        assert len(jumptimes)==len(jumpsizes)
        self.jumpsizes=jumpsizes
        self.jumptimes=jumptimes
        self.idxjumps=idx
    def sample_increment(self, DTYPEf_t initpos = 0.0, DTYPEf_t dt= 1.0):
        cdef DTYPEf_t rate = self.convergence_rate
        cdef DTYPEf_t mu = self.longterm_mean
        cdef DTYPEf_t sigma =self.sigma
        cdef DTYPEf_t exp_adt = exp(-rate*dt)
        cdef DTYPEf_t rad = sqrt((1.0-exp(-2*rate*dt))/(2*rate))
        cdef DTYPEf_t newpos =\
                mu+(initpos-mu)*exp_adt+sigma*rad*np.random.normal()
        return newpos
    def simulate(self, DTYPEf_t t0=0.0, DTYPEf_t t1=-1.0):
        if t1<=t0:
            t0=self.data[0,0]
            t1=self.data[len(self.data)-1,0]
        idx_startpos=bisect.bisect_right(self.data[:,0],t0)
        idx_timewindow=np.logical_and(self.data[:,0]>=t0, self.data[:,1]<=t1)
        cdef DTYPEf_t rate = self.convergence_rate
        cdef DTYPEf_t mu = self.longterm_mean
        cdef DTYPEf_t sigma =self.sigma
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = self.dt[idx_timewindow[:-1]]
        cdef np.ndarray[DTYPEf_t, ndim=1] exp_adt = np.exp(-rate*dt)
        cdef np.ndarray[DTYPEf_t, ndim=1] rad = np.sqrt((1.0-np.exp(-2*rate*dt))/(2*rate))
        cdef int N=len(dt), n=0
        cdef np.ndarray[DTYPEf_t, ndim=1] nz = np.random.normal(size=(N,))
        cdef np.ndarray[DTYPEf_t, ndim=2] x = np.zeros((1+N,2), dtype=DTYPEf)
        x[:,0]=np.array(self.data[idx_timewindow,0], copy=True, dtype=DTYPEf)
        x[0,1] = self.data[idx_startpos,1]
        for n in range(N):
            x[n+1,1]=mu+(x[n,1]-mu)*exp_adt[n] + sigma*rad[n]*nz[n]
        self.simulated_sample = x
    def plot(self,t0=0,t1=None):
        if t1==None:
            t1=self.data[len(self.data)-1,0]
        idx1 = np.logical_and(self.data[:,0]>=t0, self.data[:,0]<=t1)
        idx2 = np.logical_and(self.simulated_sample[:,0]>=t0, self.simulated_sample[:,0]<=t1)
        fig=plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        ax.plot(self.data[idx1,0], self.data[idx1,1], color='blue', label='data')
        ax.plot(self.simulated_sample[idx2,0], self.simulated_sample[idx2,1], color='green', label='simulation')
        ax.set_xlabel('time')
        ax.set_ylabel('log-price')
        ax.legend()
        plt.show()
            
class FourierSeries:
    def __init__(self,truncation=None,horizon=None):
        if horizon != None:
            self.set_horizon(float(horizon))
        if truncation != None:
            self.set_truncation(int(truncation))
        pass
    def set_horizon(self, DTYPEf_t T):
        self.horizon=T
    def set_truncation(self,int N=0):
        self.truncation = max(N,-N)
    def get_data(self,np.ndarray[DTYPEf_t, ndim=2] data):
        "It is assumed that data refers to log-prices, although this fact is not used anywhere"
        print('FourierSeries.get_data: shape=({},{})'.format(data.shape[0],data.shape[1]))
        if not np.isclose(data[0,0],0.0):
            print("WARNING: FourierSeries.getdata: time does not start from 0")
        cdef DTYPEf_t hor = data[len(data)-1,0]
        self.set_horizon(hor)
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = np.diff(data[:,0],append=hor)
        self.time_increment = dt
        self.data=data
        self.symmetrise()
    def symmetrise(self,):
        cdef np.ndarray[DTYPEf_t, ndim=1] tt =\
                np.concatenate([self.data[:,0], self.horizon+self.data[:,0]], axis=0)
        cdef np.ndarray[DTYPEf_t, ndim=1] xx =\
                np.concatenate([self.data[:,1], self.data[::-1,1]],axis=0)
        cdef np.ndarray[DTYPEf_t, ndim=2] symdata = \
                np.concatenate([
                    np.expand_dims(tt,axis=1),np.expand_dims(xx, axis=1)
                    ],axis=1)
        cdef DTYPEf_t TT=tt[len(tt)-1]
        cdef DTYPEf_t P = TT - tt[0]
        cdef np.ndarray[DTYPEf_t, ndim=1] dt = np.diff(tt,append=TT)
        self.P = P
        self.doubled_horizon=TT
        self.doubled_dt=dt
        self.symdata=symdata
    def store_fourier_coef(self):
        cdef int N = self.truncation
        cdef np.ndarray[DTYPEc_t, ndim=1] coef = np.array([self.compute_coef(n) for n in range(-N,N+1)])
        self.arr_coef = coef
        df_coef = pd.DataFrame({'index': range(-N,N+1), 'coef': coef})
        self.df_coef = df_coef
    def compute_coef(self,int n=0):
        cdef DTYPEf_t T = self.P
        return np.sum(self.symdata[:,1]*np.exp(-2*np.pi*1j*n*self.symdata[:,0]/T)*self.doubled_dt)/T
    def eval_series(self,DTYPEf_t t, return_real=False):
        cdef int N = self.truncation
        cdef DTYPEf_t T = self.P
        cdef np.ndarray[DTYPEc_t, ndim=1] kernel = np.array(
                [np.exp(2*np.pi*1j*n*t/T) for n in range(-N,N+1)]
        )
        if return_real:
            return np.sum(self.arr_coef*kernel).real
        else:
            return np.sum(self.arr_coef*kernel)
    def store_series_on_interval(self, int num_pnts = 100):
        cdef np.ndarray[DTYPEf_t, ndim=1] grid = np.linspace(self.data[0,0],self.horizon,num=num_pnts,dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=1] s =\
                np.array([self.eval_series(t,return_real=True) for t in list(grid)],dtype=DTYPEf)
        cdef np.ndarray[DTYPEf_t, ndim=2] series_val =\
                np.concatenate([np.expand_dims(grid, axis=1),np.expand_dims(s, axis=1)], axis=1)
        self.series_val = series_val
    def plot(self):
        fig = plt.figure(figsize=(15,8))
        ax=fig.add_subplot(111)
        ax.plot(self.data[:,0],self.data[:,1],color='blue',label='data')
        ax.plot(self.series_val[:,0],self.series_val[:,1],color='green',label='fourier_approx')
        ax.set_xlabel('time')
        ax.set_ylabel('log-price')
        ax.legend()
        plt.show()
