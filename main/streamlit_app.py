#=====================================================#
import streamlit as st
import plotly.tools as tls
import plotly.graph_objects as go
import plotly.express as px
from main_estimate import *
#=====================================================#

#? ---
#? cd main
#? streamlit run streamlit_app.py
#? ---

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open(Path(__file__).parents[0] / "style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#=====================================================#
#=====================================================#

st.sidebar.header('Model parameters')

st.sidebar.info("Classifier can take up to 1 minute")

country = st.sidebar.selectbox('Country', ('EU27','France','Germany','Spain'),index=None) 
unclass = st.sidebar.selectbox('Show unclassified', (True,False), index=0)
year = st.sidebar.selectbox('Chart start year date:',[i for i in range(2023,2000,-1)],index=8)

st.sidebar.markdown('''---''')     

order = st.sidebar.selectbox('VAR order', ("auto","fixed"), index=0)
if order=="auto":
    st.sidebar.markdown('<p style="font-size:80%;">Max lag search was set at 24</p>', unsafe_allow_html=True) 
elif order=="fixed":
    order = st.sidebar.text_input('Input VAR desired order:')
    try:
        order = int(order)
        assert 1<=order<=24
    except:
        order = None
        st.sidebar.error('Enter valid integer') #icon="🚨"
        
robust = st.sidebar.selectbox('Robust methods', (True,False), index=1)       
if robust==True:
    #st.sidebar.markdown('''---''') 
    shap_rob_plot = st.sidebar.selectbox('Plot Shapiro robust method', ['j1','j2','j3','param'], index=3)
else:
    shap_rob_plot = None
 
st.sidebar.markdown('''
---
Methodology at [SVAR3A](https://github.com/PierreRlld/SVAR3A/blob/main/notes/main.pdf).

Sources: [Shapiro(2022)](https://drive.google.com/file/d/1V-4nZikSTcfL4jZQLtwjEOLEiDDSluKD/view),
[Sheremirov(2022)](https://www.bostonfed.org/publications/current-policy-perspectives/2022/are-the-demand-and-supply-channels-of-inflation-persistent.aspx),
[ECB(box 7)](https://www.ecb.europa.eu/pub/economic-bulletin/html/eb202207.en.html)
''')
#=====================================================#
#=====================================================#

def plot_stack(df,col=None,method="shapiro",robust=None,unclassified:bool=True,year:int=2015):
            
    def yrindex(ind):
        x = []
        for el in ind:
            if el.month==1:
                x.append(el.year)
            else:
                x.append("")
        return x

    if method=="shapiro":
        color = ['cornflowerblue','indianred']
        if robust==None:
            #cols = self.meth["base"].copy()
            cols = col["base"].copy()
        else:
            #cols = self.meth[robust].copy()
            cols = col[robust].copy()
        leg = ['Demand','Supply']
    elif method=="perso":
        cols = ["dem_pers","sup_pers","dem_trans","sup_trans","dem_abg","sup_abg"]
        leg = ["Persistent demand","Persistent supply","Transitory demand","Transitory supply","Ambiguous demand","Ambiguous supply"]
        color = ['mediumseagreen','brown','royalblue','orange',"yellowgreen","darksalmon"]
    else:
        if robust=="complex":
            cols = ["dem_pers","sup_pers","dem_trans","sup_trans"]
            leg = ["Persistent demand","Persistent supply","Transitory demand","Transitory supply"]
            color = ['mediumseagreen','brown','royalblue','orange']
        else:
            cols = ["dem","sup"]
            leg = ['Demand','Supply']
            color = ['cornflowerblue','indianred']

    if unclassified==True:
        cols.append('unclassified')
        leg.append('Unclassified')
        color.append('darkgray')        
    f = df[df.index.year>=year][cols+['total']]
    f = f.rename(columns={cols[i]:leg[i] for i in range(len(cols))})
    f = f.reset_index()
    fig = px.bar(f,y=leg,x="index",labels={"index":"year","value":"%YoY HICP"},color_discrete_sequence=color)
    fig.add_trace(trace=go.Scatter(x=f["index"],y=f["total"], name="Total",mode='lines+markers', line_color="black"))
    #https://stackoverflow.com/questions/58188816/change-line-color-in-plotly
    return fig

def correlation(df):
    fig, ax = plt.subplots()
    mask = np.triu(np.ones_like(df, dtype=bool),k=1)
    sns.heatmap(df, annot=True, fmt='.2f', cmap="BuPu", linewidths=0.3, vmax=1, mask=mask, ax=ax)
    return fig

# Mise en cache
@st.cache_resource
def cpi_classifier(country,order,robust):
    if country!=None:
        meta = CPIframe(df_q_index=df_q_index, df_p_index=df_p_index, df_w=df_w, country=str(country))
        if order != None:
            cpi = CPIlabel(meta=meta,order=order,shap_robust=robust)
    return cpi

#=====================================================#
#=====================================================#
st.markdown('# HICP CLASSIFICATION')

if country==None:
    st.error('Select country',icon="🚨")

if country!=None:
    try:
        cpi = cpi_classifier(country=country,order=order,robust=robust)
        
        #*---(Proposed classification)
        st.markdown('### Proposed classification')
        perso_base = plot_stack(df=cpi.perso,method="perso",unclassified=unclass,year=year)
        st.plotly_chart(perso_base,use_container_width=True)
        #st.dataframe(data=cpi.perso)
        
        #*---(Baseline)
        st.markdown('### Baseline Shapiro classification')
        if order=="auto":
            #plot AIC selected model
            shapiro_base = plot_stack(df=cpi.shapiro_aic,col=cpi.meth,method="shapiro",robust=None,unclassified=unclass,year=year)
        else:
            shapiro_base = plot_stack(df=cpi.shapiro,col=cpi.meth,method="shapiro",robust=None,unclassified=unclass,year=year)
        st.plotly_chart(shapiro_base,use_container_width=True)
        
        st.markdown('### Baseline Sheremirov classification')
        sherem_base = plot_stack(df=cpi.sheremirov,method="sheremirov",robust=None,unclassified=unclass,year=year)
        st.plotly_chart(sherem_base,use_container_width=True)
        
        #*---(Cross)
        st.markdown('### Cross-correlations')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('#### Demand side')
            dem_corr = correlation(df=cpi.demand_corr_v)
            st.pyplot(dem_corr)
        with c2:
            st.markdown('#### Supply side')
            sup_corr = correlation(df=cpi.supply_corr_v)
            st.pyplot(sup_corr)
        
        if robust:
            #*---(Robust)
            st.markdown('### Robust Shapiro classification '+"["+shap_rob_plot+"]")
            if order=="auto":
                #plot AIC selected model
                shapiro_r = plot_stack(df=cpi.shapiro_aic_r,col=cpi.meth,method="shapiro",robust=shap_rob_plot,unclassified=unclass,year=year)
            else:
                shapiro_r = plot_stack(df=cpi.shapiro_r,col=cpi.meth,method="shapiro",robust=shap_rob_plot,unclassified=unclass,year=year)
                
            st.plotly_chart(shapiro_r,use_container_width=True)
            st.markdown('### Robust Sheremirov classification')
            sherem_r = plot_stack(df=cpi.sheremirov,method="sheremirov",robust="complex",unclassified=unclass,year=year)
            st.plotly_chart(sherem_r,use_container_width=True)
            st.markdown('### Proposed classification')
            st.plotly_chart(perso_base,use_container_width=True)
            
            
    except:
        st.error('Error in code',icon="🚨")

