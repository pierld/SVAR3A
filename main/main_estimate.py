#%%
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, acf
import numpy as np
import matplotlib.pyplot as plt
import sys


#%%
#df_q = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_log")
#df_p = pd.read_excel("data/data_flat.xlsx",sheet_name="P_log")

# === Data (index = HICP & Qt without log transf.) ===
df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")
df_q_index = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
df_p_index = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")

#df_q.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
#df_p.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
#df_p = df_p.drop(['2023-10'],axis=1)
df_q_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index = df_p_index.drop(['2023-10'],axis=1)
df_w.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)

def log_transform(cell_value):
    try:
        return np.log(float(cell_value))
    except (ValueError, TypeError):
        return cell_value

#%%
#TODO: /////////////
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
        #self.df_q = df_q_index.applymap(log_transform)
        #self.df_p = df_p_index.applymap(log_transform)
        #self.df_w = df_w
        self.qt = self.framing(df=df_q_index,transform=True)    #Log-transformed
        self.price = self.framing(df=df_p_index,transform=True) #Log-transformed
        self.qt_index = self.framing(df=df_q_index)
        self.price_index = self.framing(df=df_p_index)
        self.weights = self.framing(df=df_w)
        self.sectors = dict(self.qt.loc['HICP'])
        
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
        
    def sector(self,col_num,indexx=False,transform=True):
        """
        Generates a [price,quantity] table for the specified sector.
        Parameters:
            col_num: column number in `price` & `qt` objects for the sector of interest.
            indexx: if False takes data from log-dataset (self.price & self.qt)
            transform: (requires indexx=False)
                    if True: 
                    1. 100*first_diff 
                    2. Demean
        Returns:
            pandas.DataFrame: Dataframe of Quantity series data with dates as rows
        """
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
    
#* Shapiro labeling > cf. SVAR_Shapiro_exp.pdf pour l'explication
#* Sheremirov labeling
#TODO: rolling window estimation
class sector_estimation:
    def __init__(self,meta,col,transform=True,order="auto",maxlag=24,trend="n",sheremirov_window=[1,11]):
        """
        Args:
            meta: CPIframe object
            col: sector column number in [0,93]
        """
        self.meta = meta
        self.col = col
        self.sector = meta.sector(col_num=col,transform=transform) #DataFrame from CPIframe.sector(...,transform=true) > demeaned
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
        #* Shapiro
        self.aic.shapiro_complete = self.shapiro_label(model_resid=self.aic.resid)
        self.bic.shapiro_complete = self.shapiro_label(model_resid=self.bic.resid)
        self.aic.shapiro = self.aic.shapiro_complete[['sup+','sup-','dem+','dem-']]
        self.bic.shapiro = self.bic.shapiro_complete[['sup+','sup-','dem+','dem-']]
        #* Sheremirov
        self.sheremirov_window = sheremirov_window
        self.sheremirov_complete = self.sheremirov_label(transitory=self.sheremirov_window)
        self.sheremirov = self.sheremirov_complete[["dem","sup","dem_pers","dem_trans","sup_pers","sup_trans"]]
        
    def run_estimate(self):
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
        - Expected comovements can be infered from reduced-form residuals via sign restrictions.
            - cf. SVAR_shapiro_exp.pdf
        """
        x = model_resid.copy()
        x[['sup+','sup-','dem+','dem-']] = ""
        # demand shock
        x['dem+'] = np.where((x['qt']>0) & (x['price']>0),1,0)
        x['dem-'] = np.where((x['qt']<0) & (x['price']<0),1,0)
        # supply shock
        x['sup+'] = np.where((x['qt']>0) & (x['price']<0),1,0)
        x['sup-'] = np.where((x['qt']<0) & (x['price']>0),1,0)
        x.index = x.index.map(self.sector_dates)                #changes index from int to original date
        return x
    
    def sheremirov_label(self,transitory):
        """
        Adapted from Sheremirov (2022):
        - Uses deviation of growth rates of HICP & Demand proxy compared to 
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
        x["temp"] = x["dem"].rolling(11).sum()
        x["dem_pers"] = np.where((x['dem']==1) & (x['temp']>=transitory[1]),1,0)
        x["sup_pers"] = np.where((x['dem']==0) & (x['temp']<=transitory[0]),1,0)
        x["dem_trans"] = np.where((x['dem']==1) & (x['temp']<transitory[1]),1,0)
        x["sup_trans"] = np.where((x['dem']==0) & (x['temp']>transitory[0]),1,0)
        x = x.drop("temp",axis=1)
        return x

     
#TODO: boucle sur l'ensemble des secteurs (tester si assez de données) puis extraire résidus des estimation > fichier excel 
#%%
#? =====================================================================
eu = CPIframe(df_q_index=df_q_index, df_p_index=df_p_index, df_w=df_w, country="EU27")
names = eu.price.loc['HICP']
d1 = eu.sector(56,transform=True)
d2 = eu.sector(56,False,False)

#? =====================================================================
#%%
etest = sector_estimation(meta=eu,col=56)
#etest.aic.shapiro
#etest.aic.shapiro_complete
#etest.bic.shapiro
#etest.sheremirov
#etest.sheremirov_complete

"""
test = etest.sector_index.copy()
test['dates'] = test.index.str.split('-').str[0].astype(int)
test
test2 = test[(test['dates']<=2019)&(test['dates']>=2000)]
test2 = test2.drop("dates",axis=1)
test2
test2.mean()
test = test.drop("dates",axis=1)
test = test - test2.mean()

"""




#%%
"""
for i in range(len(eu.price().columns)):
    x = eu.process(i).dropna()
    try:
        print(i," : ",x.index[0]," - ",x.index[-1]," -- ",len(x)," -- ",names[i])
        c+=1
        #if "2022" or "2023" in x.index[-1]:
        #    if "2000" in x.index[0]:
        #        print(i," : ",x.index[0]," - ",x.index[-1]," -- ",len(x)," -- ",names[i])
        #        c+=1
    except:
        print(i,' error')
print(str(c)+" variables")

#print(estimation_test.aic.plot_acorr(25))
#restest = etest.aic.resid
#test = eu.sector(56,False)[['price']].copy()
#cycle, trend = sm.tsa.filters.hpfilter(test, 1600*3**4)

"""