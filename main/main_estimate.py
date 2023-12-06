#%%
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, acf
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Union


#%%
# === Data (index = HICP & Qt without log transf.) ===
df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")
df_q_index = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
df_p_index = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")
#```
df_q_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index = df_p_index.drop(['2023-10'],axis=1)
df_w.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)

def log_transform(cell_value):
    try:
        return np.log(float(cell_value))
    except (ValueError, TypeError):
        return cell_value

# ===============================================================
#* Shapiro labeling > cf. SVAR_Shapiro_exp.pdf pour l'explication
#* Sheremirov labeling
#* Shapiro smooth(1-3)
#TODO: télécharger overall HICP monthly

#TODO ///sector_estimation///
#TODO: Shapiro robustness : Parametric weights

#TODO ///CPIlabel///

# ===============================================================

#%%
#! //////////////////////////
class CPIframe:
    def __init__(self, df_q_index, df_p_index, df_w, country):
        """
        Args:
            df_q_index (DataFrame): The Quantity data (index).
            df_p_index (DataFrame): The Price data (index).
            df_w (DataFrame): The respective Weights data.
            country (str): Location available (EU27,France,Germany,Spain).
        """
        self.country = country
        self.qt = self.framing(df=df_q_index,transform=True)    #Log-transformed
        self.price = self.framing(df=df_p_index,transform=True) #Log-transformed
        self.qt_index = self.framing(df=df_q_index)
        self.price_index = self.framing(df=df_p_index)
        self.weights = self.framing(df=df_w)
        self.sectors = dict(self.qt.loc['HICP'])
        self.dates = list(self.price.index)[1::]
        
    def framing(self,df,transform=False):
        """
        Generates a subset of the dataframe `df` based on the specified `country`.
        'df' = df_p_index or df_q_index or df_w > from excel data_flat
        Parameters:
            df: pandas.DataFrame
            transform: log transform if True
        Returns:
            pandas.DataFrame: Dataframe of Quantity or Price series data with dates as rows.
        """
        if transform==True:
            temp = df.applymap(log_transform)
        else:
            temp = df.copy()
        x = temp[temp['Location']==self.country]
        x.reset_index(drop=True,inplace=True)
        x = x.drop(['Location'],axis=1)
        x = x.transpose()
        return x
        
    def sector(self,col_num,indexx=False,transform=True,weights=False):
        """
        Generates a [price,quantity] table for the specified sector.
        Parameters:
            col_num: column number in `price` & `qt` objects for the sector of interest.
            indexx: if False takes data from log-dataset (self.price & self.qt)
            transform: (requires indexx=False)
                    if True: 
                    1. 100*first_diff 
                    2. Demean
            weights: if True returns subset of self.weights for col_num
        Returns:
            pandas.DataFrame: Dataframe of Quantity series data with dates as rows
        """
        if not weights:
            if indexx==False:
                x = pd.concat([self.price[col_num],self.qt[col_num]],axis=1)
                sec = list(x.loc['HICP'])[-1]
                x = x.drop(['HICP'],axis=0)
                x.columns = ['price','qt']
                x = x.apply(pd.to_numeric)
                
                if transform:
                    x = 100*x.diff() 
                    x = x - x.mean()
                x = x.dropna()
                x.coicop = sec
                x.col = col_num
                return x
            
            else:
                x = pd.concat([self.price_index[col_num],self.qt_index[col_num]],axis=1)
                sec = list(x.loc['HICP'])[-1]
                x = x.drop(['HICP'],axis=0)
                x.columns = ['price','qt']
                x = x.apply(pd.to_numeric)
                
                x = x.dropna()
                x.coicop = sec
                x.col = col_num
                return x
        
        else:
            x = self.weights[[col_num]]
            sec = list(x.loc['HICP'])[-1]
            x = x.drop(['HICP'],axis=0)
            x.coicop = sec
            return x
    
    def flag_sectors(self):
        """
        Diagnostic of the different sectors.
        """
        names = self.price.loc['HICP']
        for i in range(len(self.price.columns)):
            x = self.sector(col_num=i).dropna()
            if len(x)!=0:
                if "2023" in x.index[-1]:
                    if "2000" or "2001" in x.index[0]:
                        print(i," : OK ",x.index[0],";",x.index[-1]," -- ",len(x)," obs. -- ",names[i])
                else:
                    print(i," : end missing -",x.index[0],";",x.index[-1]," -- ",names[i])
            else:
                if len(self.qt[i].dropna()) == 1 and len(self.price[i].dropna()) > 1:
                    print(i,' : PB QT ','-- price :',self.price[i].dropna().index[1],';',self.price[i].dropna().index[-1]," -- ",names[i])
                elif len(self.price[i].dropna()) == 1 and len(self.qt[i].dropna()) > 1:
                    print(i,' : PB PRICE','-- qt :',self.qt[i].dropna().index[1],';',self.qt[i].dropna().index[-1]," -- ",names[i])
                else:
                    print(i," : PB price & qt -- ",names[i])


#! //////////////////////////
class sector_estimation:
    def __init__(self,meta:CPIframe,col:int,
                 transform:bool=True,
                 order:Union[int, str]="auto",maxlag=24,trend="n",
                 shapiro:bool=True,
                 shapiro_robustness:bool=False,
                 sheremirov:bool=True,
                 sheremirov_window:list=[1,11]):
        """
        Args:
            meta: CPIframe object
            col: sector column number in [0,93]
                => If too litte data for sector 'col' then raises ValueError
            transform: xxx
            order: xxx
            maxlag: xxx 
            trend: xxx
            sheremirov_window: xxx 
        """
        self.meta = meta    #CPIframe object
        self.col = col
        self.sector = meta.sector(col_num=col,transform=transform) #DataFrame from CPIframe.sector(...,transform=true) > demeaned
        #! Flag sector
        if len(self.sector) <= 24:
            raise ValueError("Too little data for sector {0}".format(self.col))
        #````
        self.sector_w = meta.sector(col_num=col,weights=True)
        self.sector_dates = {i:x for i,x in enumerate(list(self.sector.index))}
        self.sector_index = meta.sector(col_num=col, indexx=True)
        self.order = order
        self.maxlag = maxlag
        self.trend = trend
        #````
        self.sector_ts = self.sector.reset_index(drop=True)
        self.model = VAR(endog=self.sector_ts)
        self.estimation = self.run_estimate()
        self.aic = self.estimation['aic']
        self.bic = self.estimation['bic']
        #````
        #? Shapiro
        if shapiro == True:
            self.aic.shapiro = self.shapiro_label(model_resid=self.aic.resid)
            self.bic.shapiro = self.shapiro_label(model_resid=self.bic.resid)
            if shapiro_robustness == True:
                self.aic.shapiro_smooth = self.shapiro_smooth(model_resid=self.aic.resid)
                self.bic.shapiro_smooth = self.shapiro_smooth(model_resid=self.bic.resid)
        #? Sheremirov
        if sheremirov == True:
            self.sheremirov_window = sheremirov_window
            self.sheremirov = self.sheremirov_label(transitory=self.sheremirov_window)
        
    def run_estimate(self):
        """
        Does not need to be called outside of the class definition.
        Output is a dictionary {'aic':model.fit,'bic':model.fit} where model.fit 
        """
        model_fit = {'aic':None,'bic':None}
        if self.order == "auto":
            for crit in model_fit.keys():
                model_fit[crit] = self.model.fit(maxlags=self.maxlag,ic=crit,trend=self.trend)
        else:
            for crit in model_fit.keys():
                model_fit[crit] = self.model.fit(self.order,trend=self.trend)
        return model_fit

    def shapiro_label(self,model_resid):
        """
        Adapted from Shapiro (2022):
        - Uses residuals of a reduced-form VAR
            - Input requires stationary series
            - Data processing: 
                - 1)[log(Price_index),log(Qt_index)] 
                - 2) 100*(first_diff) 
                - 3) Demean
            - Final series should be +/- stationary
        > Expected comovements can be infered from reduced-form residuals via sign restrictions.
            - cf. SVAR_shapiro_exp.pdf
        """
        x = model_resid.copy()
        x[["dem","sup","dem+","dem-","sup+","sup-"]] = ""
        # demand shock
        x['dem+'] = np.where((x['qt']>0) & (x['price']>0),1,0)
        x['dem-'] = np.where((x['qt']<0) & (x['price']<0),1,0)
        # supply shock
        x['sup+'] = np.where((x['qt']>0) & (x['price']<0),1,0)
        x['sup-'] = np.where((x['qt']<0) & (x['price']>0),1,0)
        x['dem'] = x['dem+'] + x['dem-']
        x['sup'] = x['sup+'] + x['sup-']
        x = x.drop(['price','qt'],axis=1)
        x.index = x.index.map(self.sector_dates)     #changes index from int to original date
        x = x.reindex(self.meta.dates)               #uses all dates
        for col in x.columns:
            x[col] = x[col].astype('Int8')
        return x

    def shapiro_smooth(self,model_resid):
        """
        Adapted from Shapiro (2022):
        - Smoothed labeling method.
        """
        x = model_resid.copy()
        for j in range(4):  
            #j=0 is baseline shapiro
            temp = x[['price','qt']].copy()
            temp[["temp_p","temp_q"]] = temp[['price','qt']].rolling(j+1).sum()
            temp = temp.dropna()
            temp["dem"] = np.where(((temp['temp_p']>0) & (temp['temp_q']>0)) | ((temp['temp_p']<0) & (temp['temp_q']<0)),1,0)
            temp["sup"] = np.where(((temp['temp_p']>0) & (temp['temp_q']<0)) | ((temp['temp_p']<0) & (temp['temp_q']>0)),1,0)
            if j==0:
                x[["dem","temp"]] = temp[["dem","sup"]]
            else:
                x[["dem_j{0}".format(j),"sup_j{0}".format(j)]] = temp[["dem","sup"]]
        x = x.drop(['price','qt'],axis=1)
        x.index = x.index.map(self.sector_dates)     #changes index from int to original date
        x = x.reindex(self.meta.dates)
        for col in x.columns:
            x[col] = x[col].astype('Int8')
        return x
       
    def sheremirov_label(self,transitory):
        """
        Adapted from Sheremirov (2022):
        - Uses deviation of growth rates of HICP & Demand proxy compared to 2000-19 mean
        """
        x = self.sector_index.copy()
        x = 100*x.pct_change(12).dropna()
        #``` Remove 2000-2019 mean
        x['yr'] = x.index.str.split('-').str[0].astype(int)
        m = x[(x['yr']<=2019)&(x['yr']>=2000)].drop("yr",axis=1).mean()
        x = x.drop('yr',axis=1)
        x = x - m
        #```
        x[["dem","sup","dem_pers","dem_trans","sup_pers","sup_trans"]] = ""
        x["dem"] = np.where((x['qt']>0) & (x['price']>0),1,0)
        x["sup"] = 1 - x["dem"]
        x["temp"] = x["dem"].rolling(12).sum()
        x["dem_pers"] = np.where((x['dem']==1) & (x['temp']>=transitory[1]),1,0)
        x["sup_pers"] = np.where((x['dem']==0) & (x['temp']<=transitory[0]),1,0)
        x["dem_trans"] = np.where((x['dem']==1) & (x['temp']<transitory[1]),1,0)
        x["sup_trans"] = np.where((x['dem']==0) & (x['temp']>transitory[0]),1,0)
        x = x.drop(['price','qt','temp'],axis=1)
        x = x.reindex(self.meta.dates)
        for col in x.columns:
            x[col] = x[col].astype('Int8')
        return x


#! //////////////////////////
class CPIlabel:
    def __init__(self,method,robustness=None):
        """
        Args:
            meta: CPIframe object
            method: 
            robustness:
        => If too litte data for sector 'col' then raises ValueError
        """

     
#%%
#? =====================================================================
eu = CPIframe(df_q_index=df_q_index, df_p_index=df_p_index, df_w=df_w, country="EU27")
names = eu.price.loc['HICP']
d1 = eu.sector(56,transform=True)
d12 = eu.sector(56,transform=False)
d2 = eu.sector(4,transform=True)
d3 = eu.sector(90,weights=True)
etest = sector_estimation(meta=eu,col=64,shapiro_robustness=True)

#etest.aic.shapiro
#etest.aic.shapiro_df
#etest.bic.shapiro
#etest.sheremirov
#etest.sheremirov_complete

#%%
eu.flag_sectors()


#%%
#!Flag sectors with too little data
for i in range(len(eu.price.columns)):
    x = eu.sector(col_num=i).dropna()
    if len(x)!=0:
        if "2023" in x.index[-1]:
            if "2000" or "2001" in x.index[0]:
                print(i," : ",x.index[0]," - ",x.index[-1]," -- ",len(x)," -- ",names[i])
        else:
            print(i," : LATE -",x.index[0]," - ",x.index[-1]," -- ",len(x)," -- ",names[i])
    else:
        #print(i,'PB -- qt ',len(eu.qt[i].dropna()), ' -- price ',len(eu.price[i].dropna())," -- ",names[i])
        if len(eu.qt[i].dropna()) == 1 and len(eu.price[i].dropna()) > 1:
            print(i,'PB QT ','-- price :',eu.price[i].dropna().index[1],' - ',eu.price[i].dropna().index[-1]," -- ",names[i])
        elif len(eu.price[i].dropna()) == 1 and len(eu.qt[i].dropna()) > 1:
            print(i,'PB PRICE','-- qt :',eu.qt[i].dropna().index[1],' - ',eu.qt[i].dropna().index[-1]," -- ",names[i])
        else:
            print(i,"PB price & qt -- ",names[i])


#%%
#print(estimation_test.aic.plot_acorr(25))
#restest = etest.aic.resid
#test = eu.sector(56,False)[['price']].copy()
#cycle, trend = sm.tsa.filters.hpfilter(test, 1600*3**4)

