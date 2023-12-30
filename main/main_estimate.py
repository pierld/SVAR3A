#%%
# === Dependencies ===
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, acf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from statistics import NormalDist
from tqdm import tqdm
import matplotlib
from pathlib import Path
#%%
# === Data (index = HICP & Qt without log transf.) ===
"""
#df_w = pd.read_excel("data/data_flat.xlsx",sheet_name="weights")
#df_q_index = pd.read_excel("data/data_flat.xlsx",sheet_name="QT_index")
#df_p_index = pd.read_excel("data/data_flat.xlsx",sheet_name="P_index")
#df_overall = pd.read_excel("data/overall_hicp.xlsx",sheet_name="overall")
"""

df_w = pd.read_csv(Path(__file__).parents[0] / 'df_w.csv')
df_q_index = pd.read_csv(Path(__file__).parents[0] / "df_q_index.csv")
df_p_index = pd.read_csv(Path(__file__).parents[0] / "df_p_index.csv")
df_overall = pd.read_csv(Path(__file__).parents[0] / "df_overall.csv")
df_w = df_w.drop(['Unnamed: 0'],axis=1)
df_q_index = df_q_index.drop(['Unnamed: 0'],axis=1)
df_p_index = df_p_index.drop(['Unnamed: 0'],axis=1)
df_overall = df_overall.drop(['Unnamed: 0'],axis=1)
#```
df_q_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_p_index.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_w.replace("European Union - 27 countries (from 2020)","EU27",inplace=True)
df_overall.rename(columns={"European Union - 27 countries (from 2020)":"EU27"}, inplace=True)
df_overall.set_index('Dates',inplace=True)
df_overall.index = pd.to_datetime(df_overall.index)
df_p_index = df_p_index.drop(['2023-10'],axis=1)

def log_transform(cell_value):
    try:
        return np.log(float(cell_value))
    except (ValueError, TypeError):
        return cell_value    

# ===============================================================
# ===============================================================
# ===============================================================
#%%
#! ////////////////////////////////////////////////////
class CPIframe:
    def __init__(self, df_q_index, df_p_index, df_w, country, start_flag=[2000,2001],end_flag=2023):
        """
        Args:
        - `df_q_index` (DataFrame): The Quantity data (index)
        - `df_p_index` (DataFrame): The Price data (index)
        - `df_w` (DataFrame): The respective Weights data
        - `country` (str): Location choice (EU27,France,Germany,Spain)
        - `start_flag`
        - `end_flag`
        """
        self.country = country
        self.qt = self.framing(df=df_q_index,transform=True)    #Log-transformed
        self.price = self.framing(df=df_p_index,transform=True) #Log-transformed
        self.qt_index = self.framing(df=df_q_index)
        self.price_index = self.framing(df=df_p_index)
        self.weights = self.framing(df=df_w)
        self.sectors = dict(self.qt.loc['HICP'])
        self.dates = pd.to_datetime(list(self.price.index)[1::])
        self.inflation = self.inflation()
        self.start_flag = start_flag
        self.end_flag = end_flag
        self.flag = self.flag_sector(start=self.start_flag, end=self.end_flag)
        global df_overall
        self.overall = 100*df_overall.dropna().pct_change()[[self.country]]
        
    def framing(self,df,transform=False,variation=False):
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
            temp = df.map(log_transform)
        elif variation==True:
            temp = df.map(log_transform)
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
                x.index = pd.to_datetime(x.index,format="%Y-%m")
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
                x.index = pd.to_datetime(x.index,format="%Y-%m")
                return x
        
        else:
            x = self.weights[[col_num]]
            sec = list(x.loc['HICP'])[-1]
            x = x.drop(['HICP'],axis=0)
            x.coicop = sec
            x.index = pd.to_datetime(x.index,format="%Y")
            return x
    
    def inflation(self):
        x = pd.DataFrame()
        for col in self.price_index.columns:
            temp = self.price_index[[col]]
            if len(temp.drop(['HICP'],axis=0).dropna())==0:
                temp = temp.drop(['HICP'],axis=0)
                temp.index = pd.to_datetime(temp.index)
                x[col] = temp
            else:
                temp = temp.drop(['HICP'],axis=0).dropna()
                temp.index = pd.to_datetime(temp.index)
                try:
                    temp = 100*temp.pct_change().dropna() #monthly rate in %
                    temp = temp.reindex(self.dates)
                    x[col] = temp
                except:
                    print(col)
        return x
    
    def sector_inf(self,col_num,drop=True):
        x = pd.DataFrame()
        x['inflation'] = self.inflation[[col_num]]
        x.index = pd.to_datetime(x.index,format="%Y")
        x["temp"] = x.index.year
        try:
            x["weight"] = x["temp"].apply(lambda x: self.weights.loc[str(x),col_num])
        except:
            x["weight"] = x["temp"].apply(lambda x: self.weights.loc[x,col_num])
        x = x.drop("temp",axis=1)
        x["infw"] = x["inflation"]*x["weight"]/1000
        if drop==True:
            x = x.drop(['inflation','weight'],axis=1)
        return x

    def flag_sector(self,start,end):
        """
        - Checks if 'enough' data for each sector
        """
        L = len(self.price.columns)
        flag = {x:None for x in range(L)}
        for i in range(L):
            x = self.sector(col_num=i)
            if len(x)!=0:
                if end==x.index[-1].year:
                    if any(yr==x.index[0].year for yr in start):
                        flag[i] = 0
        flag = {key: value if value is not None else 1 for key, value in flag.items()}
        return flag
        
    def flag_summary(self):
        """
        Summary of flag sector.
        """
        names = self.price.loc['HICP']
        for i in range(len(self.price.columns)):
            x = self.sector(col_num=i)
            if len(x)!=0:
                if self.end_flag==x.index[-1].year:
                    if any(yr==x.index[0].year for yr in self.start_flag):
                        print(i," : OK ",x.index[0].year,"-",x.index[0].month,";",x.index[-1].year,"-",x.index[-1].month," -- ",len(x)," obs. -- ",names[i])
                else:
                    print(i," : end missing -",x.index[0].year,"-",x.index[0].month,";",x.index[-1].year,"-",x.index[-1].month," -- ",names[i])
            else:
                if len(self.qt[i].dropna()) == 1 and len(self.price[i].dropna()) > 1:
                    print(i,' : PB QT ','-- price :',self.price[i].dropna().index[1],';',self.price[i].dropna().index[-1]," -- ",names[i])
                elif len(self.price[i].dropna()) == 1 and len(self.qt[i].dropna()) > 1:
                    print(i,' : PB PRICE','-- qt :',self.qt[i].dropna().index[1],';',self.qt[i].dropna().index[-1]," -- ",names[i])
                else:
                    print(i," : PB price & qt -- ",names[i])


#! ////////////////////////////////////////////////////
class sector_estimation:
    def __init__(self,meta:CPIframe,col:int,
                 order:Union[int, str]="auto",maxlag=24,trend="n",
                 shapiro:bool=True,
                 shapiro_robust:bool=False,
                 sheremirov:bool=True,
                 sheremirov_window:list=[1,11],
                 classify_inflation=True):
        """
        Args:
            - `meta`: `CPIframe` object
            - `col`: sector column number in [0,93]
            - If too litte data for sector 'col' then raises ValueError
            - VAR model is built with first diff then demeand log-transformed data
            - `classify_inflation`: if False only returns Shapiro and Sheremirov classification in binary form. Otherwise returns 1(dem)*weight*inf_rate / 1(sup)*w*inf_rate
                
            `VAR parametrization`
            order: if "auto" VAR order is automatically selected. Else requires an integer
            maxlag: higher bound of order selection
            trend: should remain "n" because data have been demeaned and are supposed to be stationary (no trend)
            
            `Labeling methods`
            shapiro: if True computes baseline Shapiro(2022) labeling method with reduced-form estimated VAR
            shapiro_robustness: if True also computes alternative labeling methodologies
            sheremirov: if True computes baseline Sheremirov(2022) labeling method
            sheremirov_window: [Transitory,Persistent] parametrization of step classification algo step5 
        """
        self.meta = meta    
        self.col = col
        
        #* VAR model will use log-transformed first diff and demeaned data 
        self.sector = meta.sector(col_num=col,transform=True)
        self.inflation = meta.sector_inf(col_num=col,drop=True)
        #!-----
        self.classify_inflation = classify_inflation            
        
        #! Flag sector
        if len(self.sector) <= 24:
            raise ValueError("Too little data for sector {0}".format(self.col))
        #*-----
        self.sector_w = meta.sector(col_num=col,weights=True)
        self.sector_dates = {i:x for i,x in enumerate(list(self.sector.index))}
        self.sector_index = meta.sector(col_num=col, indexx=True)
        self.order = order
        self.maxlag = maxlag
        self.trend = trend
        self.sec_name = self.meta.sectors[self.col]
        #*-----
        if self.order!="auto" and type(self.order)!=int:
            raise ValueError('Order should be set to "auto" or entered as an integer')
        #*-----
        self.sector_ts = self.sector.reset_index(drop=True)
        self.model = VAR(endog=self.sector_ts)
        self.estimation = self.run_estimate()
        #*-----
        if self.order=="auto":
            self.aic = self.estimation['aic']
            self.bic = self.estimation['bic']
            #? Shapiro
            if shapiro:
                self.aic.shapiro = self.shapiro_label(model_resid=self.aic.resid)
                self.bic.shapiro = self.shapiro_label(model_resid=self.bic.resid)
                if shapiro_robust:
                    self.aic.shapiro_robust = self.shapiro_robust(model_resid=self.aic.resid)
                    self.bic.shapiro_robust = self.shapiro_robust(model_resid=self.bic.resid)
        else:
            self.estimate = self.estimation['fixed']
            #? Shapiro
            if shapiro:
                self.estimate.shapiro = self.shapiro_label(model_resid=self.estimate.resid)
                if shapiro_robust:
                    self.estimate.shapiro_robust = self.shapiro_robust(model_resid=self.estimate.resid)
        #? Sheremirov
        if sheremirov:
            self.sheremirov_window = sheremirov_window
            self.sheremirov = self.sheremirov_label(transitory=self.sheremirov_window)
        
    def run_estimate(self):
        """
        Does not need to be called outside of the class definition.
        Output is a dictionary {'aic':model.fit,'bic':model.fit} where model.fit 
        """
        model_fit = {'aic':None,'bic':None, 'fixed':None}
        if self.order == "auto":
            for crit in ['aic','bic']:
                model_fit[crit] = self.model.fit(maxlags=self.maxlag,ic=crit,trend=self.trend)
        else:
            model_fit["fixed"] = self.model.fit(self.order,trend=self.trend)
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
            - cf. `SVAR_shapiro_exp.pdf`
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
        #changes index from int to original date & uses all dates
        x.index = x.index.map(self.sector_dates)
        x = x.reindex(self.meta.dates)               
        if self.classify_inflation==False:
            #To decerase storage size
            for col in x.columns:
                x[col] = x[col].astype('Int8')
        else:
            for col in x.columns:
                x[col] = x[col]*self.inflation['infw']
        return x

    def shapiro_robust(self,model_resid):
        """
        Adapted from Shapiro (2022):
        - Smoothed labeling method
        - Parametric weights
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
                x[["dem","sup"]] = temp[["dem","sup"]]
            else:
                x[["dem_j{0}".format(j),"sup_j{0}".format(j)]] = temp[["dem","sup"]]
        x["lambda"] = x['price']*x['qt']
        x["lambda"] = x["lambda"]*(1/x["lambda"].std())
        x["dem_param"] = x["lambda"].apply(lambda x: NormalDist().cdf(x))
        x["sup_param"] = 1 - x["dem_param"]
        x = x.drop(['price','qt','lambda'],axis=1)
        #changes index from int to original date & uses all dates
        x.index = x.index.map(self.sector_dates)     
        x = x.reindex(self.meta.dates)
        if self.classify_inflation==False:  
            #To decerase storage size
            for col in x.columns:
                if col not in ["dem_param","sup_param"]:
                    x[col] = x[col].astype('Int8')
        else:
            for col in x.columns:
                x[col] = x[col]*self.inflation['infw']
        return x
       
    def sheremirov_label(self,transitory):
        """
        Adapted from Sheremirov (2022):
        - Uses deviation of growth rates of HICP & Demand proxy compared to 2000-19 mean
        """
        x = self.sector_index.copy()
        x = 100*x.pct_change(12).dropna()
        #``` Remove 2000-2019 mean
        x['yr'] = x.index.year
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
        if self.classify_inflation==False:
            for col in x.columns:
                x[col] = x[col].astype('Int8')
        else:
            for col in x.columns:
                x[col] = x[col]*self.inflation['infw']
        return x


#! ////////////////////////////////////////////////////
#! ////////////////////////////////////////////////////
class CPIlabel:
    def __init__(self,meta:CPIframe,
                 order:Union[int, str]="auto",maxlag=24,
                 shap_robust:bool=True,
                 sheremirov_window:list[int,int]=[1,11],
                 annual_rate:bool=True):
        """
        Args:
            - `meta`: `CPIframe` object
            - `order`: if "auto" VAR order is automatically selected. Else requires an integer
            - `maxlag`: higher bound of order selection (if order="auto")
            - `shap_robust`: if True also computes alternative labeling methodologies
            - `sheremirov_window: [Transitory,Persistent] parametrization of step classification algo step5 
            - `annual_rate`: output dataframes are YoY rates, otherwise MoM
            -  NB1: VAR models are built with first diff then demeand log-transformed data

        Attributes:        
        self.demand_corr_v : cross-correlations b/w different methods for demand-driven contribution
        self.supply_corr_v : same for supply
        
        #Shapiro classification
        > if self.order is "auto"
            - shapiro_aic : overall CPI classified (demand/supply & sign)
            - shapiro_aic_sec : dataframe with all sectors classifications
            - shapiro_bic : same with BIC criterion selected model
            - shapiro_bic_sec : ---
            > if shap_robust is True
                - shapiro_aic_r : shapiro_aic but includes alternative methodologies
                - shapiro_aic_sec_r : shapiro_aic_r but includes alternative methodologies
                - shapiro_bic_r : same with BIC criterion selected model
                - shapiro_bic_r_share : ---
        > self.order is an integer
        Same objects but only one VAR of specified order was estimated for each sector (if not flagged), so no AIC/BIC
            - shapiro
            - shapiro_sec 
            > if shap_robust is True:
                - shapiro_r 
                - shapiro_sec_r
                
        #Sheremirov classification
        - sheremirov : overall CPI classified (demand/supply & sign)
        - sheremirov_sec : ---

        Methods:
        # .correlation(comp="supply" or "demand") for heatmap
        # .stack_plot(df,unclassified:bool=True,year:int=2015)
            - df : should be one of self. shapiro_aic/shapiro_bic/shapiro/sheremirov
            - unclassified : show unclassified inflation or not
            - year : starting year on plot
        """
        self.meta=meta
        self.meth={}
        self.order=order
        self.maxlag=maxlag
        self.shap_robust=shap_robust
        self.sheremirov_window=sheremirov_window
        self.annual=annual_rate
        self.demand_corr_v = pd.DataFrame()
        self.supply_corr_v = pd.DataFrame()
        #*-----
        self.sheremirov = pd.DataFrame()
        self.sheremirov_sec = None
        if self.order=="auto":
            self.shapiro_aic = pd.DataFrame()
            self.shapiro_bic = pd.DataFrame()
            self.shapiro_aic_sec = None
            self.shapiro_bic_sec = None
            if shap_robust:
                self.shapiro_aic_r = pd.DataFrame()
                self.shapiro_bic_r = pd.DataFrame()
                self.shapiro_aic_sec_r = None
                self.shapiro_bic_sec_r = None
        else:
            self.shapiro = pd.DataFrame()
            self.shapiro_sec = None
            if shap_robust:
                self.shapiro_r = pd.DataFrame()
                self.shapiro_sec_r = None
                
        self.meth["base"] = ['dem','sup']
        if shap_robust:
            self.meth["j1"] = ['dem_j1','sup_j1']
            self.meth["j2"] = ['dem_j2','sup_j2']
            self.meth["j3"] = ['dem_j3','sup_j3']
            self.meth["param"] = ['dem_param','sup_param']
        #*-----        
        self.CPIdec = self.CPI_decompose()
        self.demand_corr_v = self.demand_corr_v.corr()
        self.supply_corr_v = self.supply_corr_v.corr()
    
    def CPI_decompose(self):
        print('>> CPI decomposition for {0} processing'.format(self.meta.country))
        
        def retrieve_dates(df):
            l = len(df.columns)
            ind = df.index
            m = None
            n = None
            for i in range(len(ind)):
                if m == None:
                    if len(df.loc[ind[i]].dropna())==l and len(df.loc[ind[i+1]].dropna())==l:
                        m = ind[i]
                else:
                    if len(df.loc[ind[i]].dropna())==l:
                        n = ind[i]
            return[m,n]
        #*list(eu.dates).index(retrieve_dates(test2)[0]) #et 1
        
        def rename_col_corr(col,mode):
            if mode!=None:
                if "dem" in col:
                    return("dem_shapiro_"+mode+"_"+col.split("_")[1])
                else:
                    return("sup_shapiro_"+mode+"_"+col.split("_")[1])
            else:
                if "dem" in col:
                    return("dem_shapiro_"+col.split("_")[1])
                else:
                    return("sup_shapiro_"+col.split("_")[1])
        
        #! marche pas encore bien
        '''
        def pers_trans_perso(self,df,K=3):
            """
            df = output of sector_estimation
            """
            dc = {}
            for key in self.meth.keys():
                if key!="param":
                    for typ in ["pers","trans","abg"]:
                        dc[f"{key}_{typ}"]=[f"{prefix}_{typ}" for prefix in self.meth[key]]
            cols = [item for sub in dc.keys() for item in dc[sub]]
                        
            x = pd.DataFrame(index=df.index,columns=cols)
            dfc = df.copy().dropna()
            
            for key in self.meth.keys():
                if key!="param":
                    temp = dfc[self.meth[key]]!=0
                    #?Persistent & transitory
                    for i in range(K,len(temp)-K):
                        x.loc[dfc.index[i]][dc[f"{key}_pers"]] = temp.iloc[i-K:i+K+1][self.meth[key]].sum()>=2*K
                        x.loc[dfc.index[i]][dc[f"{key}_trans"]] = (x.loc[dfc.index[i]][dc[f"{key}_pers"]]==False)
                    #?Ambiguous
                    Lt = len(temp)-1
                    for i in range(0,K):
                        x.loc[dfc.index[Lt-i]][dc[f"{key}_abg"]] = temp.iloc[Lt-i-K::][self.meth[key]].sum()>=K+i
                        x.loc[dfc.index[Lt-i]][dc[f"{key}_trans"]] = (x.loc[dfc.index[Lt-i]][dc[f"{key}_abg"]]==False).values * (temp.loc[dfc.index[Lt-i]][self.meth[key]]).values
            x = x.reindex(t1.meta.dates)
            for col in x.columns:
                x[col] = x[col]*t1.inflation['infw']
            return x.dropna(how="all")
        '''
                
        c = []
        L_col = len(self.meta.price.columns)
        duo = ['Sector','Component']
        if self.order=="auto":
            temp_shapiro_aic = {}
            temp_shapiro_bic = {}
            if self.shap_robust:
                temp_shapiro_aic_r = {}
                temp_shapiro_bic_r = {}
        else:
            temp_shapiro = {}
            if self.shap_robust:
                temp_shapiro_r = {}
        temp_sheremirov = {}

        #*-----ESTIMATION
        with tqdm(total=L_col, ascii=True) as pbar:
            for col in range(0,L_col):
                if self.meta.flag[col]==1:
                    # Sector was flagged as missing some data
                    pass
                else:
                    c.append(col)   
                    #!---
                    estimator = sector_estimation(meta=self.meta, col=col, 
                                                  order=self.order, maxlag=self.maxlag, 
                                                  shapiro_robust=self.shap_robust, 
                                                  sheremirov_window=self.sheremirov_window, 
                                                  classify_inflation=True)
                    #!---
                    if self.order=="auto":                    
                        temp_shapiro_aic[col] = estimator.aic.shapiro
                        temp_shapiro_bic[col] = estimator.bic.shapiro
                        if self.shap_robust:
                            temp_shapiro_aic_r[col] = estimator.aic.shapiro_robust
                            temp_shapiro_bic_r[col] = estimator.bic.shapiro_robust
                    else:
                        temp_shapiro[col] = estimator.estimate.shapiro
                        if self.shap_robust:
                            temp_shapiro_r[col] = estimator.estimate.shapiro_robust
                    temp_sheremirov[col] = estimator.sheremirov
                pbar.update(1)
                
        #?-----SHAPIRO(2022)
        if self.order=="auto":              
            self.shapiro_aic_sec = pd.concat([x.transpose().stack() for x in temp_shapiro_aic.values()], keys=c, names=['Sector']).unstack()
            self.shapiro_aic_sec = self.shapiro_aic_sec.reindex(sorted(self.shapiro_aic_sec.columns),axis=1).transpose()
            self.shapiro_bic_sec = pd.concat([x.transpose().stack() for x in temp_shapiro_bic.values()], keys=c, names=['Sector']).unstack()
            self.shapiro_bic_sec = self.shapiro_bic_sec.reindex(sorted(self.shapiro_bic_sec.columns),axis=1).transpose()
            self.shapiro_aic_sec.columns.names = duo
            self.shapiro_bic_sec.columns.names = duo
            sample_aic = retrieve_dates(self.shapiro_aic_sec)
            sample_bic = retrieve_dates(self.shapiro_bic_sec)
            cols_shapiro = list(self.shapiro_aic_sec[0].columns)
            
            #*--(1)
            for col in cols_shapiro:
                self.shapiro_aic[col] = self.shapiro_aic_sec.loc[:, self.shapiro_aic_sec.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_aic[0]:sample_aic[1]]
                self.shapiro_bic[col] = self.shapiro_bic_sec.loc[:, self.shapiro_bic_sec.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_bic[0]:sample_bic[1]]
            if self.annual:
                self.shapiro_aic = self.shapiro_aic.rolling(12).sum().dropna()
                self.shapiro_bic = self.shapiro_bic.rolling(12).sum().dropna()
                self.shapiro_aic["total"] = self.meta.overall.rolling(12).sum().loc[self.shapiro_aic.index]
                self.shapiro_bic["total"] = self.meta.overall.rolling(12).sum().loc[self.shapiro_bic.index]
            else:
                self.shapiro_aic["total"] = self.meta.overall.loc[self.shapiro_aic.index]
                self.shapiro_bic["total"] = self.meta.overall.loc[self.shapiro_bic.index]
            self.shapiro_aic["unclassified"] = self.shapiro_aic["total"] - self.shapiro_aic["dem"] - self.shapiro_aic["sup"]
            self.shapiro_bic["unclassified"] = self.shapiro_bic["total"] - self.shapiro_bic["dem"] - self.shapiro_bic["sup"]

            corr_aic = self.shapiro_aic[['dem','sup']].copy().rename(columns={"dem":"dem_shapiro_aic","sup":"sup_shapiro_aic"})
            corr_bic = self.shapiro_bic[['dem','sup']].copy().rename(columns={"dem":"dem_shapiro_bic","sup":"sup_shapiro_bic"})
            self.demand_corr_v = pd.concat([corr_aic[[col for col in corr_aic.columns if "dem" in col]],corr_bic[[col for col in corr_bic.columns if "dem" in col]]],axis=1,join='inner')
            self.supply_corr_v = pd.concat([corr_aic[[col for col in corr_aic.columns if "sup" in col]],corr_bic[[col for col in corr_bic.columns if "sup" in col]]],axis=1,join='inner')
            
            #*--(2)
            if self.shap_robust:
                self.shapiro_aic_sec_r = pd.concat([x.transpose().stack() for x in temp_shapiro_aic_r.values()], keys=c, names=['Sector']).unstack()
                self.shapiro_aic_sec_r = self.shapiro_aic_sec_r.reindex(sorted(self.shapiro_aic_sec_r.columns),axis=1).transpose()
                self.shapiro_bic_sec_r = pd.concat([x.transpose().stack() for x in temp_shapiro_bic_r.values()], keys=c, names=['Sector']).unstack()
                self.shapiro_bic_sec_r = self.shapiro_bic_sec_r.reindex(sorted(self.shapiro_bic_sec_r.columns),axis=1).transpose()
                self.shapiro_aic_sec_r.columns.names = duo
                self.shapiro_bic_sec_r.columns.names = duo
                sample_aic_r = retrieve_dates(self.shapiro_aic_sec_r)
                sample_bic_r = retrieve_dates(self.shapiro_bic_sec_r)
                cols_shapiro_r = list(self.shapiro_aic_sec_r[0].columns)

                for col in cols_shapiro_r:
                    self.shapiro_aic_r[col] = self.shapiro_aic_sec_r.loc[:, self.shapiro_aic_sec_r.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_aic_r[0]:sample_aic_r[1]]
                    self.shapiro_bic_r[col] = self.shapiro_bic_sec_r.loc[:, self.shapiro_bic_sec_r.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_bic_r[0]:sample_bic_r[1]]
                
                if self.annual:
                    self.shapiro_aic_r = self.shapiro_aic_r.rolling(12).sum().dropna()
                    self.shapiro_bic_r = self.shapiro_bic_r.rolling(12).sum().dropna()
                    self.shapiro_aic_r["total"] = self.meta.overall.rolling(12).sum().loc[self.shapiro_aic_r.index]
                    self.shapiro_bic_r["total"] = self.meta.overall.rolling(12).sum().loc[self.shapiro_bic_r.index]
                else:
                    self.shapiro_aic_r["total"] = self.meta.overall.loc[self.shapiro_aic_r.index]
                    self.shapiro_bic_r["total"] = self.meta.overall.loc[self.shapiro_bic_r.index]
                self.shapiro_aic_r["unclassified"] = self.shapiro_aic_r["total"] - self.shapiro_aic_r["dem"] - self.shapiro_aic_r["sup"]
                self.shapiro_bic_r["unclassified"] = self.shapiro_bic_r["total"] - self.shapiro_bic_r["dem"] - self.shapiro_bic_r["sup"]
                    
                corr_aic_r = self.shapiro_aic_r[[col for col in self.shapiro_aic_r.columns if "_" in col]].copy()
                corr_aic_r.rename(columns={col:rename_col_corr(col,mode="aic") for col in corr_aic_r.columns},inplace=True)
                corr_bic_r = self.shapiro_bic_r[[col for col in self.shapiro_bic_r.columns if "_" in col]].copy()
                corr_bic_r.rename(columns={col:rename_col_corr(col,mode="bic") for col in corr_bic_r.columns},inplace=True)
                self.demand_corr_v = pd.concat([self.demand_corr_v,corr_aic_r[[col for col in corr_aic_r.columns if "dem" in col]],corr_bic_r[[col for col in corr_bic_r.columns if "dem" in col]]],axis=1,join='inner')
                self.supply_corr_v = pd.concat([self.supply_corr_v,corr_aic_r[[col for col in corr_aic_r.columns if "sup" in col]],corr_bic_r[[col for col in corr_bic_r.columns if "sup" in col]]],axis=1,join='inner')
                
                
        else:
            self.shapiro_sec = pd.concat([x.transpose().stack() for x in temp_shapiro.values()], keys=c, names=['Sector']).unstack()
            self.shapiro_sec = self.shapiro_sec.reindex(sorted(self.shapiro_sec.columns),axis=1).transpose()
            self.shapiro_sec.columns.names = duo
            sample_estimate = retrieve_dates(self.shapiro_sec)
            cols_shapiro = list(self.shapiro_sec[0].columns)
            
            #*--(1)
            for col in cols_shapiro:
                self.shapiro[col] = self.shapiro_sec.loc[:, self.shapiro_sec.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_estimate[0]:sample_estimate[1]]
            if self.annual:
                self.shapiro = self.shapiro.rolling(12).sum().dropna()
                self.shapiro["total"] = self.meta.overall.rolling(12).sum().loc[self.shapiro.index]
            else:
                self.shapiro["total"] = self.meta.overall.loc[self.shapiro.index]
            self.shapiro["unclassified"] = self.shapiro["total"] - self.shapiro["dem"] - self.shapiro["sup"]
            
            corr_estime = self.shapiro[['dem','sup']].copy().rename(columns={"dem":"dem_shapiro","sup":"sup_shapiro"})
            self.demand_corr_v = corr_estime[[col for col in corr_estime.columns if "dem" in col]].copy()
            self.supply_corr_v = corr_estime[[col for col in corr_estime.columns if "sup" in col]].copy()
            
            #*--(2)
            if self.shap_robust:
                self.shapiro_sec_r = pd.concat([x.transpose().stack() for x in temp_shapiro_r.values()], keys=c, names=['Sector']).unstack()
                self.shapiro_sec_r = self.shapiro_sec_r.reindex(sorted(self.shapiro_sec_r.columns),axis=1).transpose()
                self.shapiro_sec_r.columns.names = duo
                sample_estimate_r = retrieve_dates(self.shapiro_sec_r)
                cols_shapiro_r = list(self.shapiro_sec_r[0].columns)
                
                #*--(2.1)
                for col in cols_shapiro_r:
                    self.shapiro_r[col] = self.shapiro_sec_r.loc[:, self.shapiro_sec_r.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_estimate_r[0]:sample_estimate_r[1]]
                
                #*--(2.2)
                if self.annual:
                    self.shapiro_r = self.shapiro_r.rolling(12).sum().dropna()
                    self.shapiro_r["total"] = self.meta.overall.rolling(12).sum().loc[self.shapiro_r.index]
                else:
                    self.shapiro_r["total"] = self.meta.overall.loc[self.shapiro_r.index]
                self.shapiro_r["unclassified"] = self.shapiro_r["total"] - self.shapiro_r["dem"] - self.shapiro_r["sup"]
                
                corr_estime_r = self.shapiro_r[[col for col in self.shapiro_r.columns if "_" in col]].copy()
                corr_estime_r.rename(columns={col:rename_col_corr(col,mode=None) for col in corr_estime_r.columns},inplace=True)
                self.demand_corr_v = pd.concat([self.demand_corr_v,corr_estime_r[[col for col in corr_estime_r.columns if "dem" in col]]],axis=1,join='inner')
                self.supply_corr_v = pd.concat([self.supply_corr_v,corr_estime_r[[col for col in corr_estime_r.columns if "sup" in col]]],axis=1,join='inner')

        #?-----SHEREMIROV(2022)
        self.sheremirov_sec = pd.concat([x.transpose().stack() for x in temp_sheremirov.values()], keys=c, names=['Sector']).unstack()
        self.sheremirov_sec = self.sheremirov_sec.reindex(sorted(self.sheremirov_sec.columns),axis=1).transpose()
        self.sheremirov_sec.columns.names = duo
        sample_shemirov = retrieve_dates(self.sheremirov_sec)
        cols_sheremirov = list(self.sheremirov_sec[0].columns)
        
        for col in cols_sheremirov:
            self.sheremirov[col] = self.sheremirov_sec.loc[:, self.sheremirov_sec.columns.get_level_values('Component') == col].sum(axis=1).loc[sample_shemirov[0]:sample_shemirov[1]]
        if self.annual:
            self.sheremirov = self.sheremirov.rolling(12).sum().dropna()
            self.sheremirov["total"] = self.meta.overall.rolling(12).sum().loc[self.sheremirov.index]
        else:
            self.sheremirov["total"] = self.meta.overall.loc[self.sheremirov.index]
        self.sheremirov["unclassified"] = self.sheremirov["total"] - self.sheremirov["dem"] - self.sheremirov["sup"]
        
        corr_sd_sher = self.sheremirov[["dem","sup"]].copy().rename(columns={"dem":"dem_sheremirov","sup":"sup_sheremirov"})
        self.demand_corr_v = pd.concat([self.demand_corr_v,corr_sd_sher[[col for col in corr_sd_sher.columns if "dem" in col]]],axis=1,join='inner')
        self.supply_corr_v = pd.concat([self.supply_corr_v,corr_sd_sher[[col for col in corr_sd_sher.columns if "sup" in col]]],axis=1,join='inner')
        return
    
    def correlation(self,comp:str):
        """
        Args:
            - comp is "supply" or "demand"
        """
        if comp not in ["supply","demand"]:
            return
        else:
            fig = plt.figure(figsize=(20, 9))
            if comp=="supply":
                mask = np.triu(np.ones_like(self.supply_corr_v, dtype=bool),k=1)
                sns.heatmap(self.supply_corr_v, annot=True, fmt='.2f', cmap="BuPu", linewidths=0.3, vmax=1, mask=mask)
            else:
                mask = np.triu(np.ones_like(self.demand_corr_v, dtype=bool),k=1)
                sns.heatmap(self.demand_corr_v, annot=True, fmt='.2f', cmap="BuPu", linewidths=0.3, vmax=1, mask=mask)

    def plot_stack(self,df,method="shapiro",robust=None,unclassified:bool=True,year:int=2015):
        """
        Args:
            - df : attribute of CPIlabel
                - shapiro_aic / shapiro_bic / shapiro
                - shapiro_aic_r / shapiro_bic_r / shapiro_r
                - sheremirov
            - method : "shapiro" or "sheremirov"
            - robust : in ['j1','j2','j3','param'] for "shapiro" / "complex" shows Persistent/Trans decompostion for "sheremirov"
            - unclassified : show unclassified component
            - unclassified : show total inflation
        """
                
        def yrindex(ind):
            x = []
            for el in ind:
                if el.month==1:
                    x.append(el.year)
                else:
                    x.append("")
            return x

        if method=="shapiro":
            color = ['cornflowerblue','tab:red']
            if robust==None:
                cols = self.meth["base"].copy()
            else:
                cols = self.meth[robust].copy()
            leg = ['Total','Demand','Supply']

        else:
            if robust=="complex":
                cols = ["dem_pers","dem_trans","sup_pers","sup_trans"]
                leg = ["Total","Persistent demand","Transitory demand","Persistent supply","Transitory supply"]
                color = ['mediumseagreen','royalblue','indianred','orange']
            else:
                cols = ["dem","sup"]
                leg = ['Total','Demand','Supply']
                color = ['cornflowerblue','tab:red']

        if unclassified==True:
            cols.append('unclassified')
            leg.append('Unclassified')
            color.append('darkgray')      
        fig = plt.figure(figsize=(20, 9))  
        f = df[df.index.year>year][cols+['total']]
        ind = f.index
        f = f.reset_index(drop=True)
        ax = f[cols].plot.bar(stacked=True, width = 1, color=color)
        f['total'].plot(color="black",ax=ax)
        ax.set_xticklabels(yrindex(ind))
        ax.legend(leg,loc = "upper left")
        ax.figure.set_facecolor('white')
        

#%%
#? =====================================================================
meta = CPIframe(df_q_index=df_q_index, df_p_index=df_p_index, df_w=df_w, country="France")
cpi = CPIlabel(meta=meta)
#t1 = sector_estimation(meta=meta,col=64,shapiro_robust=True)


#%%
"""
def retrieve_dates(df):
    l = len(df.columns)
    ind = df.index
    m = None
    n = None
    for i in range(len(ind)):
        if m == None:
            if len(df.loc[ind[i]].dropna())==l and len(df.loc[ind[i+1]].dropna())==l:
                m = ind[i]
        else:
            if len(df.loc[ind[i]].dropna())==l:
                n = ind[i]
    return[m,n]

duo = ['Sector', 'Component']
test = cpi_eu.shapiro_aic_sec_r
dts = retrieve_dates(df=test)
test2 = test.loc[:, test.columns.get_level_values('Component') == 'dem']
res = test2.sum(axis=1).loc[dts[0]:dts[1]]

#test = pd.MultiIndex.from_tuples([], names=duo)
#test = test.append(pd.MultiIndex.from_tuples([(64,j) for j in t1.aic.shapiro_robust], names=duo))
#multi_index = pd.MultiIndex.from_product([['aic', 'bic'], t1.aic.shapiro_robust.columns], names=duo)
#test = pd.concat([t1.aic.shapiro_robust.transpose().stack(), t1.bic.shapiro_robust.transpose().stack()], keys=['aic', 'bic'], names=['Sector']).unstack()
#df_multi_combined = pd.concat([df_multi, df_c.stack()], keys=['C'], names=['Letter']).unstack()
#test = test.transpose()
"""

