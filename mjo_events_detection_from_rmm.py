import numpy as np
import pandas as pd

def select_seg1_modf(i,len_1,len_0):
    
    ind1 = []
    ind2 = []
    z = []
    i[0] = 0
    i[-1] = 0
    t2 = i[1:]
    t3 = i[0:-1];
    z =  t2-t3;
   
    ind1 = np.where(z == 1)[0] +1;
    ind2 = np.where(z == -1 ) [0]+1 ;
    
    if (ind1[0] > ind2[0]):
        ind2 = ind2[1:]
    if (ind1[-1]>ind2[-1]):
        ind1 = ind1[:-1]
    
#     print(len(ind1))
    
    ind_1 = np.where((ind2-ind1)< len_1)[0]
    
#     print(len(ind_1))
    
    ind1 = np.delete(ind1,ind_1)
    ind2 = np.delete(ind2,ind_1)
    k = ind1[1:] - ind2[:-1]
    w = np.where(k<=len_0)[0]
    
    while len(w)>0:
        for j in w:
                ind2[j] = ind2[j+1]
        ind1 = np.delete(ind1,j+1)
        ind2 = np.delete(ind2,j+1)
        k = ind1[1:] - ind2[:-1]
        w = np.where(k<=len_0)[0]
     
   
    return ind1,ind2


def mjo_events(AA,PP,s_year,n,opt):
    
    len_1 =  15
    len_0 =  8
    sphase = 5
    ephase = 6
    ph_cov = 6
    ph = np.array([1,2,3,4,5,6,7,8])
    
    no_events  = np.zeros((n,))
    life_span  = np.zeros((n,))
    lf_567 = np.zeros((n,))
    lf_123 =np.zeros((n,))
    
    events = np.zeros((n,), dtype=np.object)
    
    n1   = s_year
    
    for i in np.arange(n):

        if opt=='allyear':
            a1 = AA.loc[str(n1)+'-01-01':str(n1)+'-12-31'].values*1
            p1 = PP.loc[str(n1)+'-01-01':str(n1)+'-12-31'].values*1
        elif opt=='summer':
            a1 = AA.loc[str(n1)+'-05-01':str(n1)+'-10-31'].values*1
            p1 = PP.loc[str(n1)+'-05-01':str(n1)+'-10-31'].values*1
        elif opt =='winter':
            a1 = AA.loc[str(n1)+'-11-01':str(n1+1)+'-04-30'].values*1
            p1 = PP.loc[str(n1)+'-11-01':str(n1+1)+'-04-30'].values*1
        
        ind1,ind2= select_seg1_modf(a1,len_1=len_1,len_0=len_0)
        
#         print(ind1,ind2)
        
        
        eve= []
        life = []
        l_567 =[]
        l_123 =[]
        
        for x,y in zip(ind1,ind2):
            
            p11 = p1[x:y]
            p22 = p11[1:] - p11[:-1]
            ll = np.where(p22<0)[0]
            
            ###  check for consecutive events ###########
            
            if len(ll)!=0:
                l1 = [0]
                l1 = l1+(ll+1).tolist()+[len(p11)]
              
                
                for n in range(len(l1)-1):
                    p33 = p11[l1[n]:l1[n+1]]
                    if len(p33.tolist())>=len_1:
                        if (p33[0]<=sphase) & (p33[-1]>=ephase):
                            if len(np.intersect1d(ph,p33))>= ph_cov :
                                eve = eve +  [p33.tolist()]
                                life = life + [len(p33)]
                                l_123 = l_123+[len(np.where(np.array(p33)<4)[0])]
                                l_567 = l_567+[len(np.where((np.array(p33)<=6)&(np.array(p33)>=4))[0])]
                            
            else:
                if (p11[0]<=sphase) & (p11[-1]>=ephase):
                    if len(np.intersect1d(ph,p11))>= ph_cov :
                        eve = eve + [p11.tolist()]
                        life = life + [len(p11)] 
                        l_123 = l_123+[len(np.where(np.array(p11)<4)[0])]
                        l_567 = l_567+[len(np.where((np.array(p11)<=6)&(np.array(p11)>=4))[0])]
           
        
        events[i]   = eve
        life_span[i] =   np.mean(np.array(life))  
        lf_123[i] =   np.mean(np.array(l_123)) 
        lf_567[i] =   np.mean(np.array(l_567)) 
#         print(events[i])
        no_events[i]= len(events[i])
        n1= n1+1
        
    return no_events,events,life_span,lf_123,lf_567
