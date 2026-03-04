
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import TheilSenRegressor
import json

# Configuration de la page
st.set_page_config(page_title="Smart-Station IA", layout="wide")

# --- PARAMÈTRES ET LIAISONS ---
# REMPLACE CES LIENS PAR TES LIENS DIRECTS ONEDRIVE
URL_CUVES = "Inventaire_Cuves_Variation.csv"
URL_POMPES = "Transactions_Cuves_Variation.csv"

LIAISONS = {1: [1, 3], 2: [2, 4]}
SEUIL_LEGAL = 0.5
ALPHA = 0.99
PROJECTION_INTERVALLES = 35040

# --- LOGIQUE DE RÉCONCILIATION (Cellule 1 & 2) ---
@st.cache_data
def charger_et_reconcilier(url_c, url_p):
    try:
        df_cuves = pd.read_csv(url_c)
        df_pompes = pd.read_csv(url_p)
        
        df_cuves['Timestamp'] = pd.to_datetime(df_cuves['Date'] + ' ' + df_cuves['Heure'], dayfirst=True)
        df_pompes['Timestamp'] = pd.to_datetime(df_pompes['Date'] + ' ' + df_pompes['Heure'], dayfirst=True)
        df_pompes['Slot'] = df_pompes['Timestamp'].dt.floor('15min') + pd.Timedelta(minutes=15)

        p_pivot = df_pompes.pivot_table(index=['ID_Cuve', 'Slot'], columns='ID_Pompe', 
                                        values='Volume_Vendu', aggfunc='sum').fillna(0).reset_index()

        df_cuves = df_cuves.sort_values(['ID_Cuve', 'Timestamp'])
        df_cuves['Baisse_Cuve'] = df_cuves.groupby('ID_Cuve')['Volume_L'].shift(1) - df_cuves['Volume_L']

        df = pd.merge(df_cuves, p_pivot, left_on=['ID_Cuve', 'Timestamp'], right_on=['ID_Cuve', 'Slot'], how='inner')
        
        cols_p = [c for c in df.columns if str(c).isdigit()]
        df['Ventes_Totales'] = df[cols_p].sum(axis=1)
        
        df = df[(df['Baisse_Cuve'] >= 0) & (df['Ventes_Totales'] > 0)].copy()
        df['Ratio_Brut'] = ((df['Baisse_Cuve'] - df['Ventes_Totales']) / df['Ventes_Totales']) * 100
        df['Ratio_Lisse'] = df.groupby('ID_Cuve')['Ratio_Brut'].transform(lambda x: x.rolling(window=8, min_periods=1).mean())
        df['Pompes_Actives'] = df[cols_p].apply(lambda row: ",".join([str(p) for p in cols_p if row[p] > 0]), axis=1)
        
        return df, cols_p
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, []

# --- LOGIQUE IA ET DIAGNOSTIC (Cellule 3) ---
def calculer_ia_complet(df, cols_p):
    pompes_ids = [int(c) for c in cols_p]
    cusum_vals = {p: 0.0 for p in pompes_ids}
    
    # 1. Calcul CUSUM (Litres)
    for idx, row in df.iterrows():
        ecart_L = row['Baisse_Cuve'] - row['Ventes_Totales']
        for p in pompes_ids:
            ratio = row[str(p)] / row['Ventes_Totales'] if row['Ventes_Totales'] > 0 else 0
            cusum_vals[p] = (ALPHA * cusum_vals[p]) + (ecart_L * ratio)
            df.at[idx, f'CUSUM_P{p}'] = cusum_vals[p]

    # 2. Isolation Forest
    model_if = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly_Score'] = model_if.fit_predict(df[['Ratio_Lisse']])
    
    rapport = []
    projections = {}

    for id_c, pompes_cuve in LIAISONS.items():
        subset = df[df['ID_Cuve'] == id_c].sort_values('Timestamp')
        
        # Anomalies brusques avec ta logique métier
        anomalies = subset[(subset['Anomaly_Score'] == -1) & (abs(subset['Ratio_Brut']) > SEUIL_LEGAL)]
        for _, r in anomalies.iterrows():
            if r['Ratio_Brut'] > 0.5:
                type_ano = "VOL / FUITE"
            elif r['Ratio_Brut'] < -0.5:
                type_ano = "MÉTROLOGIE / PRÉSENCE D'AIR"
            else:
                type_ano = "NON CLASSIFIÉ"
            rapport.append(f"🚨 {r['Timestamp'].strftime('%d/%m %H:%M')} [Cuve {id_c}] : {type_ano} ({r['Ratio_Brut']:.2f}%)")

        # 3. Régression Theil-Sen (Prédiction)
        for p_id in pompes_cuve:
            y = subset[f'CUSUM_P{p_id}'].values
            x = np.arange(len(y)).reshape(-1, 1)
            if len(y) > 20:
                model_ts = TheilSenRegressor(random_state=42)
                model_ts.fit(x, y)
                future_x = np.arange(len(y), len(y) + 100).reshape(-1, 1)
                future_y = model_ts.predict(future_x)
                projections[p_id] = (future_x, future_y)
                
    return df, rapport, projections

# --- INTERFACE (Cellule 4) ---
st.title("⛽ Smart-Station : Monitoring IA & Réconciliation")

df_final, pompes_detectees = charger_et_reconcilier(URL_CUVES, URL_POMPES)

if df_final is not None:
    df_final, journal, proj_dict = calculer_ia_complet(df_final, pompes_detectees)
    
    cuve_id = st.sidebar.selectbox("Sélectionner la Cuve", [1, 2])
    df_cuve = df_final[df_final['ID_Cuve'] == cuve_id]

    # Graphique Ratio Brut
    st.subheader(f"Analyse des Écarts - Cuve {cuve_id}")
    fig_ratio = go.Figure()
    colors = np.where(df_cuve['Ratio_Brut'].abs() <= SEUIL_LEGAL, '#00ff88', '#ff4b4b')
    fig_ratio.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=df_cuve['Ratio_Brut'], mode='markers', 
                                   marker=dict(color=colors, size=6), name="Ratio %"))
    fig_ratio.add_hline(y=SEUIL_LEGAL, line_dash="dash", line_color="red", annotation_text="Limite Fuite")
    fig_ratio.add_hline(y=-SEUIL_LEGAL, line_dash="dash", line_color="red", annotation_text="Limite Air/Métro")
    fig_ratio.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_ratio, use_container_width=True)

    # Graphiques CUSUM par Pompe
    st.subheader("Santé Métrologique & Prédictions")
    cols = st.columns(2)
    for i, p_id in enumerate(LIAISONS[cuve_id]):
        with cols[i]:
            fig_p = go.Figure()
            vol_cum = df_cuve[str(p_id)].cumsum()
            limite = vol_cum * (SEUIL_LEGAL / 100)
            fig_p.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=df_cuve[f'CUSUM_P{p_id}'], name="Erreur Cumulée", line=dict(color='#00d4ff')))
            fig_p.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=limite, name="Limite +", line=dict(color='red', dash='dot')))
            fig_p.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=-limite, name="Limite -", line=dict(color='red', dash='dot')))
            
            if p_id in proj_dict:
                fx, fy = proj_dict[p_id]
                last_t = df_cuve['Timestamp'].max()
                future_t = [last_t + pd.Timedelta(minutes=15*j) for j in range(len(fy))]
                fig_p.add_trace(go.Scatter(x=future_t, y=fy, name="Tendance IA", line=dict(color='yellow', dash='dash')))
            
            fig_p.update_layout(title=f"Pompe {p_id}", template="plotly_dark", height=350)
            st.plotly_chart(fig_p, use_container_width=True)

    # Journal des alertes
    with st.expander("Consulter le Rapport de Diagnostic IA"):
        for msg in journal:
            if f"Cuve {cuve_id}" in msg:
                st.write(msg)
