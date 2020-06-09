
# coding: utf-8

# In[1]:

import numpy as np
from scipy.stats import expon
from scipy.stats import bernoulli

def gridCeiling(grid,x):
    i=0;
    while (grid[i]<x and i<grid.shape[0]-1):
        i+=1;
    xCeiling=grid[i];
    return xCeiling

def gridFloor(grid,x):
    i=1;
    while grid[-i]>x:
        i+=1;
    xFloor=grid[-i];
    return xFloor



# In[2]:


def mergeGrids(grid1,grid2):
    #length=grid1.shape[0]+grid2.shape[0]
    #mergedGrid=np.zeros(length)
    #isGrid1=np.zeros(length,dtype=bool)
    grid=np.concatenate((grid1,grid2),axis=0)
    mergedGrid=np.array([np.min(grid)])
    isGrid1=(mergedGrid[0]==np.min(grid1))
    while (mergedGrid[-1]<np.max(grid)):
        grid1=grid1[np.nonzero(grid1>mergedGrid[-1])]
        #print('grid1={}'.format(grid1))
        grid2=grid2[np.nonzero(grid2>mergedGrid[-1])]
        grid=np.concatenate((grid1,grid2),axis=0)
        mergedGrid=np.append(mergedGrid,np.min(grid))
        #print('mergedGrid={}'.format(mergedGrid))
        if (np.any((grid1>=mergedGrid[-1]))):
            #display('enter if')
            #print('mergedGrid[-1]={}'.format(mergedGrid[-1]))
            isGrid1=np.append(isGrid1,(mergedGrid[-1]==np.min(grid1)))
        else:
            #display('enter else')
            isGrid1=np.append(isGrid1,False)
    return mergedGrid,isGrid1


# In[ ]:


def generate_exponentialTime(intensity,timeHorizon):
    waitingTime=expon.rvs(size=1,scale=1/intensity)
    lastTime=np.cumsum(waitingTime)[-1]
    while lastTime<timeHorizon:
        waitingTime=np.append(waitingTime,
                              expon.rvs(size=1,scale=1/intensity),
                              )
        lastTime+=waitingTime[-1]
    
    jumpTime=np.cumsum(waitingTime)
    return jumpTime

def randomSign(size):
    randomSign=2*(bernoulli.rvs(0.5, size=size)-0.5)
    return randomSign