import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import TheilSenRegressor

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart-Station IA Energenic", layout="wide")

def transformer_en_lien_direct(url):
    try:
        base_url = url.split("?")[0]
        return f"{base_url}?download=1"
    except Exception: return url

# Liens Dynamiques
URL_CUVES = transformer_en_lien_direct("https://energenic-my.sharepoint.com/:x:/r/personal/anicet_bibout_energenic_cloud/Documents/Documents/SMARTSTATION/ALGO/Simulateur/Inventaire_Cuves_Variation.csv?d=w62aa55394c1a40ceb273281ab619d0f2&csf=1&web=1&e=x2iV4a")
URL_POMPES = transformer_en_lien_direct("https://energenic-my.sharepoint.com/:x:/r/personal/anicet_bibout_energenic_cloud/Documents/Documents/SMARTSTATION/ALGO/Simulateur/Inventaire_Cuves_Variation.csv?d=w62aa55394c1a40ceb273281ab619d0f2&csf=1&web=1&e=x2iV4a")

LIAISONS = {1: [1, 3], 2: [2, 4]}
SEUIL_LEGAL = 0.5
PROJECTION_INTERVALLES = 35040 # 1 an en créneaux de 15min

# --- CHARGEMENT & RÉCONCILIATION ---
@st.cache_data(ttl=600)
def charger_donnees(url_c, url_p):
    try:
        df_c = pd.read_csv(url_c)
        df_p = pd.read_csv(url_p)
        df_c['Timestamp'] = pd.to_datetime(df_c['Date'] + ' ' + df_c['Heure'], dayfirst=True)
        df_p['Timestamp'] = pd.to_datetime(df_p['Date'] + ' ' + df_p['Heure'], dayfirst=True)
        df_p['Slot'] = df_p['Timestamp'].dt.floor('15min') + pd.Timedelta(minutes=15)
        
        p_pivot = df_p.pivot_table(index=['ID_Cuve', 'Slot'], columns='ID_Pompe', values='Volume_Vendu', aggfunc='sum').fillna(0).reset_index()
        df_c = df_c.sort_values(['ID_Cuve', 'Timestamp'])
        df_c['Baisse_Cuve'] = df_c.groupby('ID_Cuve')['Volume_L'].shift(1) - df_c['Volume_L']
        
        df = pd.merge(df_c, p_pivot, left_on=['ID_Cuve', 'Timestamp'], right_on=['ID_Cuve', 'Slot'], how='inner')
        cols_p = [c for c in df.columns if str(c).isdigit()]
        df['Ventes_Totales'] = df[cols_p].sum(axis=1)
        df = df[(df['Baisse_Cuve'] >= 0) & (df['Ventes_Totales'] > 0)].copy()
        df['Ratio_Brut'] = ((df['Baisse_Cuve'] - df['Ventes_Totales']) / df['Ventes_Totales']) * 100
        return df, cols_p
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None, []

# --- ANALYSE PRÉDICTIVE ---
def analyser_ia(df, cols_p):
    rapport = []
    stats_pompes = {}
    
    # Isolation Forest pour les anomalies
    model_if = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = model_if.fit_predict(df[['Ratio_Brut']])

    for id_c, pompes_cuve in LIAISONS.items():
        sub = df[df['ID_Cuve'] == id_c].sort_values('Timestamp')
        
        for p_id in pompes_cuve:
            # Calcul CUSUM
            ratio_p = sub[str(p_id)] / sub['Ventes_Totales']
            sub[f'CUSUM_P{p_id}'] = (sub['Baisse_Cuve'] - sub['Ventes_Totales']) * ratio_p
            sub[f'CUSUM_P{p_id}'] = sub[f'CUSUM_P{p_id}'].cumsum()
            
            y = sub[f'CUSUM_P{p_id}'].values
            x = np.arange(len(y)).reshape(-1, 1)
            limite_L = sub[str(p_id)].sum() * (SEUIL_LEGAL / 100)

            if len(y) > 20:
                # Régression Robuste
                model_ts = TheilSenRegressor(random_state=42)
                model_ts.fit(x, y)
                futur_x = np.arange(len(y), len(y) + PROJECTION_INTERVALLES).reshape(-1, 1)
                futur_y = model_ts.predict(futur_x)
                
                # Calcul Échéance
                idx_dep = np.where(abs(futur_y) > limite_L)[0]
                if abs(y[-1]) > limite_L:
                    status, color = "🔴 HORS-NORME", "red"
                elif len(idx_dep) > 0:
                    jours = round((idx_dep[0] * 15) / 1440, 1)
                    status, color = (f"🟡 CRITIQUE : {jours}j", "orange") if jours < 30 else (f"🟡 PRÉVISION : {jours}j", "yellow")
                else:
                    status, color = "✅ SANTÉ : OK", "#00d4ff"
                
                stats_pompes[p_id] = {"status": status, "color": color, "futur": futur_y, "limite": limite_L}

    return df, rapport, stats_pompes

# --- DASHBOARD ---
st.title("⛽ Smart-Station : Maintenance IA")
data, p_ids = charger_donnees(URL_CUVES, URL_POMPES)

if data is not None:
    data, journal, stats = analyser_ia(data, p_ids)
    c_id = st.sidebar.selectbox("Sélection Cuve", [1, 2])
    
    # Cartes d'échéance
    cols = st.columns(len(LIAISONS[c_id]))
    for i, p_id in enumerate(LIAISONS[c_id]):
        with cols[i]:
            s = stats.get(p_id, {"status": "N/A", "color": "grey"})
            st.markdown(f"**Pompe {p_id}**")
            st.subheader(f":{s['color']}[{s['status']}]")

    # Graphique de dérive (CUSUM + Projection)
    st.divider()
    for p_id in LIAISONS[c_id]:
        if p_id in stats:
            fig = go.Figure()
            df_v = data[data['ID_Cuve'] == c_id]
            fig.add_trace(go.Scatter(x=df_v['Timestamp'], y=df_v[f'CUSUM_P{p_id}'], name="Dérive Réelle (L)"))
            
            # Ajout de la projection
            futur_t = [df_v['Timestamp'].max() + pd.Timedelta(minutes=15*j) for j in range(1000)] # On affiche les 1000 prochains
            fig.add_trace(go.Scatter(x=futur_t, y=stats[p_id]['futur'][:1000], name="Projection IA", line=dict(dash='dash')))
            
            # Limites
            fig.add_hline(y=stats[p_id]['limite'], line_color="red", line_dash="dot")
            fig.add_hline(y=-stats[p_id]['limite'], line_color="red", line_dash="dot")
            
            fig.update_layout(title=f"Projection de dérive Pompe {p_id}", template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
