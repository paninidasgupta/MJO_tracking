#!/usr/bin/env python
# coding: utf-8

# ## Tracking the eastward positive precipitation anomaly
# 
# The present method follows the technique of zhang and ling(2017) to identify the MJO events tracking the precipitation anomaly using TRMM datasets.
# The method follows three basics steps

# Written by Prajeesh Ag, Scientist-D, IITM, Pune

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset,num2date
from scipy.interpolate import RectBivariateSpline
import seaborn as sns
from IPython.display import clear_output
from multiprocessing import Pool

dx            = 0.25 # (in degrees)
dt            = 1. # (in days)
vmin          = 1. # (in m/sec)
vmax          = 16. # (in m/sec)
nlines        = 60  # number of trial lines
x_org         = 89.875 # longitude origin
gap_threshold = 2. # (in degrees)
prop_lim      = 50. # (in degrees)
speed_lim     = [2,8] # (m/s)
day_lim       = 20 # (20 days)
day_lim2      = 2*day_lim

nproc         = 8

iday_lim      = int(day_lim/dt)
iday_lim2      = int(day_lim2/dt)
v_range       = np.linspace(vmin,vmax,nlines)

iprop_lim     = int(prop_lim/dx)
ispeed_lim    = np.where(np.logical_and(v_range>=speed_lim[0],v_range<=speed_lim[1]))[0][[0,-1]]

imaxtim = 1000

def update_progress(progress,prefix=""):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = prefix+" Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


####################### Function to convert meter/second to day/degree longitude or slope range##########

def convert_day_per_lon(v):  ## meter/second
    
    v_day      = v*3600.*24./(110*1000)  # degree/day
    m_day_deg  =  1./v_day   # rounding upto second decimal
    
    return m_day_deg



################################# Function to generate trial lines:#################################

def trial_lines(x_range,m_range,x_org,time):

    y_org     = time

    yfit      = np.zeros((len(x_range),len(m_range),len(y_org)))
    xfit      = np.zeros((len(x_range),len(m_range),len(y_org)))
    
    for t in np.arange(len(y_org)):  
        for i in np.arange(len(m_range)):    
            ytemp             =  m_range[i]*(x_range-x_org)+y_org[t]
            ytemp[ytemp<0.0]  = np.nan
            yfit[:,i,t]       = ytemp
            xfit[:,i,t]       = x_range
    
    return xfit, yfit
        
    
    
#######################function to retain only longest segment ###############################################

def retain_longest_segment_par(mask):
    p = Pool(nproc)
    x = p.map(retain_longest_segment, (mask[:,:,i::nproc] for i in range(nproc)))
    imask = mask.astype(int)
    segStart = (np.ones(mask.shape[1:])*-1).astype(int)
    segEnd = (np.ones(mask.shape[1:])*-1).astype(int)
    for i in range(nproc):
        imask[:,:,i::nproc] = x[i][0]
        segStart[:,i::nproc] = x[i][1]
        segEnd[:,i::nproc] = x[i][2]   
    return imask, segStart, segEnd


def retain_longest_segment(mask):
    gap = int(gap_threshold/dx)
    istrt = 0
    iend  = mask.shape[0]-1
    imask = mask.astype(int)
    segStart = (np.ones(mask.shape[1:])*-1).astype(int)
    segEnd = (np.ones(mask.shape[1:])*-1).astype(int)
#     fig, ax = plt.subplots(3,1)
    
    for t in np.arange(mask.shape[2]):
        update_progress(t/mask.shape[2],'retain_longest_segment')
        for m in np.arange(mask.shape[1]):
            # do nothing if all values are 1
            if all(mask[:,m,t]):
                segStart[m,t] = istrt
                segEnd[m,t] = iend
                continue
                
            # do nothing if all values are 0    
            if not any(mask[:,m,t]):
                continue
                
            temp = imask[:,m,t]
            df   = pd.DataFrame(list(temp),columns=['A'])
            df['block'] = (df.A.shift(1) != df.A).astype(int).cumsum()
            B = df.reset_index().groupby(['A','block'])['index'].apply(np.array)
                    
            for k in np.arange(B[0].shape[0]):
                zlocs=B[0].iloc[k]
                #do nothing if zero segment occurs at the start or at the end
                if any(zlocs==istrt) or any(zlocs==iend):
                    continue
                    
                #replace with 1 if length zero segment is less than gap
                if len(zlocs)<=gap:
                    temp[zlocs] = 1
            
            
            df   = pd.DataFrame(list(temp),columns=['A'])
            df['block'] = (df.A.shift(1) != df.A).astype(int).cumsum()
            B=df.reset_index().groupby(['A','block'])['index'].apply(np.array)
            
            
            #if only one segment is there, save it back to imask and go to next iteration
            if B[1].shape[0]==1:
                imask[:,m,t]=temp
                wtemp = np.where(temp==1)
                segStart[m,t] = wtemp[0][0]
                segEnd[m,t] = wtemp[0][-1]
                continue
            
            #if more than one segment is there, then retain only the longest segment
            zlen = len(B[1].iloc[0])
            kmax = 0
            
            for k in np.arange(1,B[1].shape[0]):
                if zlen<len(B[1].iloc[k]):
                    zlen = len(B[1].iloc[k])
                    kmax = k
                    
            zlocs = B[1].iloc[kmax]
            temp[:] = 0
            temp[zlocs] = 1
            imask[:,m,t] = temp
            wtemp = np.where(temp==1)
            segStart[m,t] = wtemp[0][0]
            segEnd[m,t] = wtemp[0][-1]
    update_progress(1,'retain_longest_segment')
    return imask, segStart, segEnd



def find_local_maxima(A,window):
    wnd2=int(window/2)
    strt=wnd2
    endd=A.shape[1]-wnd2
    Amax = np.full(A.shape,False)
    for t in np.arange(strt,endd):
        temp = A[:,t-wnd2:t+wnd2+1]
        Atemp = Amax[:,t-wnd2:t+wnd2+1]
        mval = np.amax(temp)
        Atemp[temp==mval] = True
        Amax[:,t-wnd2:t+wnd2+1] = Atemp  
    return Amax

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def subtract(self, p):
        return Point(self.x - p.x, self.y - p.y)

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

# calculates the cross product of vector p1 and p2
# if p1 is clockwise from p2 wrt origin then it returns +ve value
# if p2 is anti-clockwise from p2 wrt origin then it returns -ve value
# if p1 p2 and origin are collinear then it returs 0
def cross_product(p1, p2):
    return p1.x * p2.y - p2.x * p1.y

def direction(p1, p2, p3):
    return  cross_product(p3.subtract(p1), p2.subtract(p1))

def left(p1, p2, p3):
    return direction(p1, p2, p3) < 0

def right(p1, p2, p3):
    return direction(p1, p2, p3) > 0

def collinear(p1, p2, p3):
    return direction(p1, p2, p3) == 0

# checks if p lies on the segment p1p2
def on_segment(p1, p2, p):
    return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)

# checks if line segment p1p2 and p3p4 intersect
def intersect(seg1, seg2):
    p1 = Point(seg1[0],seg1[1])
    p2 = Point(seg1[2],seg1[3])
    p3 = Point(seg2[0],seg2[1])
    p4 = Point(seg2[2],seg2[3])
    
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and         ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False



def get_all_info(segVld,lon,time,v_range,tly,segStart,segEnd,var1,A):
    speed=[]; startlon=[]; endlon=[]; startdate=[]
    enddate=[]; zonalscale=[]; proprange=[]; lifespan=[]
    amplitude=[]; zonalscale_max=[]
    tot = len(segVld)
    k1 = 0
    for [m,t] in segVld:
        k1=k1+1
        update_progress(k1/tot,'get_all_info')
        speed.append(v_range[m])
        startlon.append(lon[segStart[m,t]])
        endlon.append(lon[segEnd[m,t]])
        proprange.append(lon[segEnd[m,t]]-lon[segStart[m,t]])
        startdate.append(num2date(tly[segStart[m,t],m,t],time.units,calendar=time.calendar))
        enddate.append(num2date(tly[segEnd[m,t],m,t],time.units,calendar=time.calendar))
        lifespan.append(tly[segEnd[m,t],m,t]-tly[segStart[m,t],m,t])
        amplitude.append(A[m,t])
        ipr_ave=0
        ipr_max=0
        npts=0
        skp=int((segEnd[m,t]-segStart[m,t])/10)
        if skp<1:
            skp = 1
        for i in np.arange(segStart[m,t],segEnd[m,t]+1,skp):
            y1 = tly[i,m,t]
            j = np.argmin(np.abs(time-y1))
            npts=npts+1
            ipr = 0
            k = i
            while k>=0:
                if var1[j,k]:
                    ipr = ipr + 1
                else:
                    break
                k=k-1
            
            k=i
            while k<lon.size:
                if var1[j,k]:
                    ipr = ipr + 1
                else:
                    break
                k=k+1
            ipr_ave = ipr_ave + ipr
            ipr_max = np.maximum(ipr_max,ipr)
        
        ipr_ave = (ipr_ave*dx)/npts
        ipr_max = ipr_max*dx
        zonalscale.append(ipr_ave)
        zonalscale_max.append(ipr_max)
        df = pd.DataFrame(list(zip(speed,startdate,enddate,lifespan,startlon,endlon,
                                  proprange,amplitude,zonalscale,zonalscale_max)),
                         columns=['speed','startdate','enddate','lifespan','startlon','endlon',
                                  'propogation_range','amplitude','zonalscale','zonalscale_max'])
    return df




def overlaping_segments_par(Epts,Mval,Tval,m1,t1,A,seg1,lon,tly,time,segStart,segEnd):
    p = Pool(nproc)
    idel = []
    valid = True
    x = p.map(overlaping_segments_wrap, ((Epts[i::nproc],Mval[i::nproc],Tval[i::nproc],
                                         m1,t1,seg1,A,lon,tly,time,segStart,segEnd,var1) 
              for i in range(nproc)))

    for i in range(nproc):
        idel = idel + (np.array(x[i][0])*nproc+i).tolist()
        valid = x[i][1] and valid

    return idel, valid

def overlaping_segments_wrap(args):
    return overlaping_segments(*args)


def overlaping_segments(Epts,Mval,Tval,m1,t1,seg1,A,lon,tly,time,segStart,segEnd,var1):
    k = -1
    valid = True
    idel = []
    iseg1 = np.arange(segStart[m1,t1],segEnd[m1,t1]+1)
    for (seg2, m2, t2) in zip(Epts,Mval,Tval):
        k = k + 1
        if ((seg2[1]-seg1[1])<iday_lim):
            if (A[m1,t1]>A[m2,t2]):
                idel.append(k)
            else:
                valid = False
                break
        elif intersect(seg1,seg2):
            if (A[m1,t1]>A[m2,t2]):

                idel.append(k)
            else:
                valid = False
                break
        else:
            iseg2 = np.arange(segStart[m2,t2],segEnd[m2,t2]+1)
            iseg  = np.intersect1d(iseg1,iseg2)
            if iseg.size:               
                segY1 = tly[iseg,m1,t1]
                segY2 = tly[iseg,m2,t2]
                if segY2[0]-segY1[0]>day_lim2 or segY2[-1]-segY1[-1]>day_lim2:
                    continue
                    
                connected=False
                skp=int(iseg.size/10)
                if skp<1:
                    skp = 1
                    
                for i in np.arange(0,iseg.size,skp):
                    sy1 = segY1[i]
                    sy2 = segY2[i]
                    iy=np.where(np.logical_and(time>=sy1,time<=sy2))[0]
                    if all(var1[iy,iseg[i]]): #single envelope
                        connected=True
                        break
                
                if connected:
                    if (A[m1,t1]>A[m2,t2]):
                        idel.append(k)
                    else:
                        valid = False
                        break
    return idel, valid


def overlaping_segments1(lEpts,lMval,lTval,A):
    
    nEpts = np.array(lEpts)
    nMval = np.array(lMval)
    nTval = np.array(lTval)
    
    isrt = nEpts[:,1].argsort()
    Epts = nEpts[isrt,:].tolist()
    Mval = nMval[isrt].tolist()
    Tval = nTval[isrt].tolist()

    while True:
        nod = 0
        for od in range(0,2):
            idel = []
            for i in range(od,len(Tval),2):
                if i==len(Tval)-1:
                    break
                seg1 = Epts[i]
                seg2 = Epts[i+1]
                m1 = Mval[i]
                m2 = Mval[i+1]
                t1 = Tval[i]
                t2 = Tval[i+1]       
                if (seg2[1]-seg1[1]<iday_lim):
                    if (A[m1,t1]>A[m2,t2]):
                        idel.append(i+1)
                    else:
                        idel.append(i)

            #print(len(idel))
            for index in sorted(idel, reverse=True):
                del Epts[index]
                del Mval[index]
                del Tval[index]
            
            #print(len(Tval))
            
            if len(idel)==0:
                nod = nod + 1
        
        if nod == 2:
            break
    
    nEpts = np.array(Epts)
    nMval = np.array(Mval)
    nTval = np.array(Tval)
    
    isrt = nTval.argsort()
    Epts = nEpts[isrt,:].tolist()
    Mval = nMval[isrt].tolist()
    Tval = nTval[isrt].tolist()
    
    #print(len(Epts))
    
    return Epts, Mval, Tval



def remove_overlaping_segments(segEndpts,segMval,segTval,lon,tly,time,segStart,segEnd,A,var1):
    
    isrt = segTval.argsort()
    segTval = segTval[isrt]
    segMval = segMval[isrt]
    segEndpts = segEndpts[isrt,:]

    Epts = segEndpts.tolist()
    Tval = segTval.tolist()
    Mval = segMval.tolist()
    
    totlen = len(Epts)
    
    idel = []
    for i in np.arange(len(Epts)):
        m1   = Mval[i]
        t1   = Tval[i]
        iseg1 = np.arange(segStart[m1,t1],segEnd[m1,t1]+1)
        if iseg1.size < iprop_lim:
            idel.append(i)
        #elif m1 < ispeed_lim[0] or m1 > ispeed_lim[1]:
        #    idel.append(i)

    for index in sorted(idel, reverse=True):
        del Epts[index]
        del Mval[index]
        del Tval[index]
    
    update_progress(1.-len(Epts)/totlen,'remove_overlaping_segments')
    print(len(Epts))
    Epts, Mval, Tval = overlaping_segments1(Epts,Mval,Tval,A)
    print(len(Epts))
    
    segVld = []
    while len(Epts)>1:
        update_progress(1.-len(Epts)/totlen,'remove_overlaping_segments')
        idel = []
        seg1 = Epts[0]
        m1   = Mval[0]
        t1   = Tval[0]
        del Epts[0]
        del Mval[0]
        del Tval[0]

        ni = np.argmax(np.array(Tval)-t1 > imaxtim)+1
        if ni==1 and (not any(np.array(Tval)-t1 > imaxtim)):
            ni = len(Tval)
        
        print(len(Epts),ni)
        idel, valid = overlaping_segments(Epts[0:ni],Mval[0:ni],Tval[0:ni],
                                            m1,t1,seg1,A,lon,tly,time,segStart,segEnd,var1)
        
        for index in sorted(idel, reverse=True):
            del Epts[index]
            del Mval[index]
            del Tval[index]

        if valid:
            segVld.append([m1,t1])

    update_progress(1,'remove_overlaping_segments')
    
    return segVld


def apply_speed_lim(segVld):
    idel = []
    i = -1
    for [m,t] in segVld:
        i = i + 1
        if m <= ispeed_lim[0] or m >= ispeed_lim[1]:
            idel.append(i)
    
    for index in sorted(idel, reverse=True):
        del segVld[index]


# In[ ]:


m_range       = convert_day_per_lon(v_range)
dataset       = Dataset('MJO_filtered.nc')
var           = dataset.variables['mjo_time_lon'][:]
lon           = dataset.variables['lon'][:]
time          = dataset.variables['time']

stddata       = Dataset('MJO_filtered_std.nc')
sigma         = stddata['mjo_time_lon'][:]

maxtim        = int(dx*lon.size*m_range[0])
imaxtim       = int(maxtim/dt)

print(maxtim)

#Initialize interpolation
interp_spline = RectBivariateSpline(time[:], lon, var)

#Initialize trial lines
tlx, tly      = trial_lines(lon,m_range,x_org,time[:])

#Do interpolation onto trial lines
P = interp_spline(tly,tlx,grid=False)

print(P.shape)


mask  = np.full(P.shape, False)

for i in np.arange(sigma.size):
    mask[i,:,:] = P[i,:,:]>=sigma[i]

imask, segStart, segEnd = retain_longest_segment_par(mask)



A = np.nansum(P*imask,0)
window=11
Amax = find_local_maxima(A,window)
print('After find local maxima!')

var1=np.zeros(var.shape)
for i in np.arange(var.shape[0]):
    var1[i,:]=(var[i,:]>=sigma)*1

segStart = segStart.astype(int)
segEnd = segEnd.astype(int)

segEndpts = np.ones((np.sum(Amax),4))
segTval = np.ones(np.sum(Amax)).astype(int)
segMval = np.ones(np.sum(Amax)).astype(int)

k=0

for t in np.arange(Amax.shape[1]):
    for m in np.arange(Amax.shape[0]):
        if not Amax[m,t]:
            continue
        segEndpts[k,0] = lon[segStart[m,t]]
        segEndpts[k,1] = tly[segStart[m,t],m,t]
        segEndpts[k,2] = lon[segEnd[m,t]]
        segEndpts[k,3] = tly[segEnd[m,t],m,t]
        segTval[k] = t
        segMval[k] = m
        k = k + 1


# In[ ]:



segVld = remove_overlaping_segments(segEndpts,segMval,segTval,lon,tly,time,segStart,segEnd,A,var1)

apply_speed_lim(segVld)

nmask=imask*1.
nmask[imask==0]=np.nan

# im=plt.contourf(lon,time[:],var,cmap='jet')

# plt.colorbar()

# for [i,j] in segVld:
#     plt.plot(tlx[:,i,j]*nmask[:,i,j],tly[:,i,j]*nmask[:,i,j],'k')
# plt.ylim(0,1000)


# plt.ylim(0,100)

infostat = get_all_info(segVld,lon,time,v_range,tly,segStart,segEnd,var1,A)

infostat.to_csv('MJO_infostat_60_lines_2gap.csv')


# In[ ]:


im=plt.contourf(lon,time[:],var,cmap='jet')

plt.colorbar()

for [i,j] in segVld:
    plt.plot(tlx[:,i,j]*nmask[:,i,j],tly[:,i,j]*nmask[:,i,j],'k')
#     plt.ylim(0,500)



#plt.ylim(5037,5150)


# In[ ]:


print(infostat)


# In[ ]:




