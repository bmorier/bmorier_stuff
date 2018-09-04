# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:29:57 2018

@author: bmori

Here we analyze data from TRTH. First anaysis is on ibov and BRL futures. 
Next I intend to do the same analysis on SPX em Treasury futures.

TODO
clean data (see method cited) - ok
robust skellam (pdf & cdf) - ok
Version with center of bid-ask
mixture pdf and cdf ( 0 + skellam + skellamtail)


Roadmap PHD thesis

Non-Linear Filtering: Extending NAIS
-Higher Dimension
-Multiple Frequencies
-Non-linear state space ?

Stochastic Vol Intraday: Including Dynamic Seasonality and Tail on the distribution
--Empirical: S&P and TY
A Stochastic Correlation Intraday Model
--Empirical: S&P and TY
-Compare with Score


"""

#srs2=pd.read_csv('out2.csv',parse_dates=[4],infer_datetime_format=True)

from pandas.tseries.offsets import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import mUtil as uu
from addict import *

fromCSV=True
doPlot=True
doTreatSeries=False

def findBadTicks(srs1):
    ret=np.diff(np.log(srs1.values))
    #r0=srs1.rolling(W,center=True)
#    mean0= r0.apply(uu.mean_t)

#    ret[isOvernight1[1:]]=ret[isOvernight1[1:]]-mean0[isOvernight1[1:]] #devolve a media de retorno, estimativa robusta do gap overnight
    #srs_ = pd.Series(np.cumsum(ret))
    srs_=srs1
    W=60
    nstd=5
    nstd0=1.5
    #nstd=3
    #nstd0=0.5

    std0=uu.std_t(ret)*W**0.5
    r=srs_.rolling(W,center=True)
    std1= r.apply(uu.std_t)
    mean1= r.apply(uu.mean_t)

    I=np.abs(srs1.values-mean1)>nstd*std1 +nstd0*std0*mean1
    
    return I


def devPlot(srs1,):
    ret=np.diff(np.log(srs1.values))
    W=60
    nstd=5
    nstd0=1.5
    std0=uu.std_t(ret)*W**0.5
    r=srs1.rolling(W,center=True)
    std1= r.apply(uu.std_t)
    mean1= r.apply(uu.mean_t)

    plt.figure();
    plt.plot(srs1.values)
    plt.plot(mean1.values)
    plt.plot((mean1- nstd*std1 -nstd0*std0*mean1).values)
    plt.plot((mean1+ nstd*std1 +nstd0*std0*mean1).values)
    
path1 = 'D:\\Users\\Bruno\\Google Drive\\_bmorier\\__Doutorado FGV\\_tese\\volatility\\data\\correl_python\\'

srsInfo = Dict()

srsInfo.ind_dol.csv='out2.csv'
srsInfo.ind_dol.pickle='ibov_usd'
srsInfo.ind_dol.srsName=['IND','DOL']
srsInfo.ind_dol.srsCode=['INDc1','DOLc1']
srsInfo.ind_dol.fields=['code','underlying','dt','gmt_off','lastPx']

srsInfo.ind_dol_ba.csv='out_east_bid_ask.csv'
srsInfo.ind_dol_ba.pickle='ibov_usd_bid_ask'
srsInfo.ind_dol_ba.srsName=['IND','DOL']
srsInfo.ind_dol_ba.srsCode=['INDc1','DOLc1']
srsInfo.ind_dol_ba.fields=['code','dt','gmt_off','lastPx','bid','ask']


srs=Dict()
if fromCSV:
    for nm in srsInfo:
        srs[nm]=pd.read_csv(path1 + srsInfo[nm].csv)
        dt1=pd.to_datetime(srs[nm]['Date-Time'],format='%Y-%m-%dT%H:%M:%S.%fZ')
        srs[nm]['Date-Time']=dt1
        
        del srs[nm]['Domain']
        del srs[nm]['Type']
        
        if not srsInfo[nm].fields is None:
            srs[nm].columns = srsInfo[nm].fields
        else:
            srs[nm].columns[:4]=['code','underlying','dt','gmt_off']
            
        srs[nm].gmt_off= srs[nm].gmt_off.astype(int)
        srs[nm].to_pickle(srsInfo[nm].pickle)

else:
    for nm in srsInfo:
        srs[nm]=pd.read_pickle(srsInfo[nm].pickle)




if doTreatSeries:

    for nm in srsInfo:
        srsTreat = srs.loc[:,:];
    
        delta1=pd.to_timedelta(srsTreat.gmt_off,unit='h');
    
        srsTreat.dt = srsTreat.dt + delta1;
        
        IND = srsTreat[srsTreat.code=='INDc1']
        DOL = srsTreat[srsTreat.code=='DOLc1']
        
        
        IND=IND.set_index('dt')
        DOL=DOL.set_index('dt')
        
        IND['dtd']=IND.index.normalize()
        IND['isOvernight'] = np.array([True,] + list(IND.dtd[1:].values!=IND.dtd[0:-1].values) )
        IND['h']=IND.index-IND.dtd
        J=IND.h<pd.Timedelta('16:00:00')
        IND=IND[J]
        
        
        yy1=np.diff(IND.lastPx)/5
        yy1=yy1[~IND['isOvernight'][1:]]
        ds=np.sort(np.abs(yy1))[::-1]
        plt.hist(ds[:2000],bins=100)
        ii=np.where(np.abs(yy1)==ds[0])[0][0]
        devPlot(IND['lastPx'][ii-10000:ii+10000])
        
        I=findBadTicks(IND['lastPx'])
        IND=IND[np.logical_not(I)]
        
        
        
        
        
        DOL['dtd']=DOL.index.normalize()
        DOL['isOvernight'] = np.array([True,] + list(DOL.dtd[1:].values!=DOL.dtd[0:-1].values) )
        DOL['h']=DOL.index-DOL.dtd
        J=DOL.h<pd.Timedelta('16:00:00')
        DOL=DOL[J]
        
        I=findBadTicks(DOL['lastPx'])
        DOL=DOL[np.logical_not(I)]
        #yy2=np.diff(DOL.lastPx)/0.5
        #yy2=yy2[~DOL['isOvernight'][1:]]
        #ds=np.sort(np.abs(yy1))[::-1]
        
        
       
        DF = IND.merge(DOL,'inner', left_index=True, right_index=True)
        DF1 = IND.merge(DOL,'outer', left_index=True, right_index=True)
        
        del DF['code_x'],DF['code_y'], DF['gmt_off_y'], DF['dtd_y'], DF['isOvernight_y'], DF['h_y']
        del DF1['code_x'],DF1['code_y'], DF1['gmt_off_y'], DF1['dtd_y'], DF1['isOvernight_y'], DF1['h_y']
        
        DF.columns = ['gmt_off','IND','dtd','isOvernight','h','DOL']
        DF1.columns = ['gmt_off','IND','dtd','isOvernight','h','DOL']
        #DF1 = ind1.merge(dol1,'outer', left_index=True, right_index=True)
        
        #DF.iloc[:5000,:].to_excel('DF.xlsx')
        #DF1.iloc[:5000,:].to_excel('DF1.xlsx')
        DF1=DF1.fillna(method='ffill')
        DF1=DF1.fillna(method='bfill')
        
        #clean
        
        DF1.to_pickle('DF1')
        DF.to_pickle('DF')
        IND.to_pickle('IND')
        DOL.to_pickle('DOL')

else:

    DF1 = pd.read_pickle('DF1')
    DF = pd.read_pickle('DF')
    IND = pd.read_pickle('IND')
    DOL = pd.read_pickle('DOL')


yIND=np.diff(DF1.IND)/5
min1,max1 = np.percentile(yIND,[0.1,99.9])
Iind = np.logical_and(yIND>=min1, yIND<=max1)

yDOL=np.diff(DF1.DOL)/0.5
min1,max1 = np.percentile(yDOL,[0.1,99.9])
Idol = np.logical_and(yDOL>=min1, yDOL<=max1)

II=np.logical_and(Iind,Idol)


y0 = gu.cat(1,[gu.ap1(yIND[II],1),gu.ap1(yDOL[II],1)])

plt.figure()
plt.hist(y0[:,0],bins=10)
plt.figure()
plt.hist(y0[:,1],bins=10)

np.save('y0',y0)


yIND=np.diff(DF.IND)/5
min1,max1 = np.percentile(yIND,[0.1,99.9])
Iind = np.logical_and(yIND>=min1, yIND<=max1)

yDOL=np.diff(DF.DOL)/0.5
min1,max1 = np.percentile(yDOL,[0.1,99.9])
Idol = np.logical_and(yDOL>=min1, yDOL<=max1)

II=np.logical_and(Iind,Idol)


y1 = gu.cat(1,[gu.ap1(yIND[II],1),gu.ap1(yDOL[II],1)])

plt.figure()
plt.hist(y1[:,0],bins=10)
plt.figure()
plt.hist(y1[:,1],bins=10)

np.save('y1',y1)

## Seasonality

#create equaly spaced series
ind1 = IND.resample('30S').last()
ind1=ind1.fillna(method='ffill')



def addDateTimeInfo(ind1):
    ind1['dtd']=ind1.index.normalize()
    ind1['isOvernight'] = np.array([True,] + list(ind1.dtd[1:].values!=ind1.dtd[0:-1].values) )
    ind1['h']=ind1.index-ind1.dtd
    return ind1

ind1=addDateTimeInfo(ind1);
J=(ind1.h<=pd.Timedelta('16:00:00')) & (ind1.h>=pd.Timedelta('9:05:00'))
ind1= ind1[J]

ind1['ret']=uu.cat(0,[np.atleast_1d(0) ,np.diff(np.log(ind1.lastPx))])


    
dol1 = DOL.resample('30S').last()
dol1=dol1.fillna(method='ffill')
dol1=addDateTimeInfo(dol1);
J=(dol1.h<=pd.Timedelta('16:00:00')) & (dol1.h>=pd.Timedelta('9:05:00'))
dol1= dol1[J]

dol1['ret']=uu.cat(0,[np.atleast_1d(0) ,np.diff(np.log(dol1.lastPx))])


def mad(x):
    return np.nanmedian(np.abs(x-np.nanmedian(x)))*1.43

def mad1(x):
    return np.nanmean(np.abs(x))
#pivot table
gmtDaily=pd.pivot_table(ind1,values='gmt_off',index='dtd')

gmtDaily1=gmtDaily.iloc[(gmtDaily==-1).values[:,0],:];

ind2=ind1.iloc[ind1.dtd.isin(gmtDaily1.index).values,:]


indH=pd.pivot_table(ind2,values='ret',index='h',aggfunc=mad1)


indH1=pd.pivot_table(ind1,values='ret',index='dtd',columns='h')

#indH=indH[ np.logical_and(indH!=0, np.logical_not(np.isnan(indH)))]
indH=indH.iloc[ ((indH!=0) & ~(pd.isnull(indH))).values[:,0] ]

indH.head()


dolH=pd.pivot_table(dol1,values='ret',index='h',aggfunc=mad1)


dolH1=pd.pivot_table(dol1,values='ret',index='dtd',columns='h')

#indH=indH[ np.logical_and(indH!=0, np.logical_not(np.isnan(indH)))]
dolH=dolH.iloc[ ((dolH!=0) & ~(pd.isnull(dolH))).values[:,0] ]

plt.plot(dolH[1:])

#IND=srs2[']

# IND:
# Quebra as 09:30 (no gmt-3) ? Pre market off ?
# Quebra as 10:00 (abertura do a vista, em todos)
#  Quebra na abrtura off (10:30 ou 11:30)
#


