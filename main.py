import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import TheilSenRegressor

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart-Station IA Energenic", layout="wide")

def transformer_drive_en_direct(url):
    try:
        if "drive.google.com" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return url
    except Exception: return url

# Liens Drive
URL_CUVES = transformer_drive_en_direct("https://drive.google.com/file/d/1BxdKjJB7Difw4vfe4OKylMV_b5PAGUgL/view?usp=sharing")
URL_POMPES = transformer_drive_en_direct("https://drive.google.com/file/d/1H19rgLxGU7wL5VRhDNg2h9_WMf8rFk9s/view?usp=sharing")

LIAISONS = {1: [1, 3], 2: [2, 4]}
SEUIL_LEGAL = 0.5
ALPHA = 1.0  # CUSUM BRUT (non lissé) pour voir toutes les variations
PROJECTION_INTERVALLES = 35040  # 1 AN EXACT

# --- CHARGEMENT ---
@st.cache_data(ttl=600)
def charger_donnees(url_c, url_p):
    try:
        df_c = pd.read_csv(url_c)
        df_p = pd.read_csv(url_p)
        df_c['Timestamp'] = pd.to_datetime(df_c['Date'] + ' ' + df_c['Heure'], dayfirst=True)
        df_p['Timestamp'] = pd.to_datetime(df_p['Date'] + ' ' + df_p['Heure'], dayfirst=True)
        df_p['Slot'] = df_p['Timestamp'].dt.floor('15min') + pd.Timedelta(minutes=15)
        
        df_p['ID_Pompe'] = df_p['ID_Pompe'].astype(str)
        p_pivot = df_p.pivot_table(index=['ID_Cuve', 'Slot'], columns='ID_Pompe', values='Volume_Vendu', aggfunc='sum').fillna(0).reset_index()
        
        df_c = df_c.sort_values(['ID_Cuve', 'Timestamp'])
        df_c['Baisse_Cuve'] = df_c.groupby('ID_Cuve')['Volume_L'].shift(1) - df_c['Volume_L']
        
        df = pd.merge(df_c, p_pivot, left_on=['ID_Cuve', 'Timestamp'], right_on=['ID_Cuve', 'Slot'], how='inner')
        cols_p = [c for c in p_pivot.columns if c not in ['ID_Cuve', 'Slot']]
        df['Ventes_Totales'] = df[cols_p].sum(axis=1)
        
        df = df[(df['Baisse_Cuve'] >= 0) & (df['Ventes_Totales'] > 0)].copy()
        df['Ratio_Brut'] = ((df['Baisse_Cuve'] - df['Ventes_Totales']) / df['Ventes_Totales']) * 100
        df['Ratio_Lisse'] = df.groupby('ID_Cuve')['Ratio_Brut'].transform(lambda x: x.rolling(window=8, min_periods=1).mean())
        return df, cols_p
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, []

# --- ANALYSE IA COMPLÈTE ---
def analyser_ia_complet(df, cols_p):
    cusum_vals = {p: 0.0 for p in cols_p}
    for idx, row in df.iterrows():
        ecart_cuve_litres = row['Baisse_Cuve'] - row['Ventes_Totales']
        for p in cols_p:
            ratio_debit = row[p] / row['Ventes_Totales'] if row['Ventes_Totales'] > 0 else 0
            cusum_vals[p] = (ALPHA * cusum_vals[p]) + (ecart_cuve_litres * ratio_debit)
            df.at[idx, f'CUSUM_P{p}'] = cusum_vals[p]

    model_if = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly_Score'] = model_if.fit_predict(df[['Ratio_Lisse']])
    
    rapport_diagnostic = []
    diagnostics_preventifs = {}

    for id_c, pompes_cuve in LIAISONS.items():
        subset = df[df['ID_Cuve'] == id_c].sort_values('Timestamp')
        if subset.empty: continue

        for p_id in pompes_cuve:
            p_str = str(p_id)
            if f'CUSUM_P{p_str}' in subset.columns:
                y = subset[f'CUSUM_P{p_str}'].values
                x = np.arange(len(y)).reshape(-1, 1)
                vol_p_total = subset[p_str].sum()
                limite_p_litres = vol_p_total * (SEUIL_LEGAL / 100)

                if len(y) > 20:
                    model_ts = TheilSenRegressor(random_state=42)
                    model_ts.fit(x, y)
                    futur_x = np.arange(len(y), len(y) + PROJECTION_INTERVALLES).reshape(-1, 1)
                    futur_y = model_ts.predict(futur_x)
                    
                    idx_dep = np.where(abs(futur_y) > limite_p_litres)[0]
                    if abs(y[-1]) > limite_p_litres:
                        msg, color, status_txt = "🔴 HORS-NORME", "red", "DÉPASSEMENT"
                    elif len(idx_dep) > 0:
                        jours = round((idx_dep[0] * 15) / 1440, 1)
                        msg, color = "🟡 PRÉVISION", "orange"
                        status_txt = f"{jours} jours restants"
                    else:
                        msg, color, status_txt = "✅ OK", "#00d4ff", "Stable > 1 an"
                    
                    diagnostics_preventifs[p_id] = {"msg": msg, "color": color, "futur": futur_y, "jours": status_txt}
                    rapport_diagnostic.append(f"📊 Pompe {p_id} [Cuve {id_c}] : {status_txt}")

    return df, rapport_diagnostic, diagnostics_preventifs

# --- DASHBOARD ---
st.title("⛽ Smart-Station IA : Maintenance & Métrologie")
data, p_ids = charger_donnees(URL_CUVES, URL_POMPES)

if data is not None:
    data, journal, stats = analyser_ia_complet(data, p_ids)
    c_id = st.sidebar.selectbox("Sélection Cuve", [1, 2])
    df_c = data[data['ID_Cuve'] == c_id].sort_values('Timestamp')

    # --- INDICATEURS HAUT DE PAGE ---
    cols_header = st.columns(len(LIAISONS[c_id]))
    for i, p_id in enumerate(LIAISONS[c_id]):
        with cols_header[i]:
            s = stats.get(p_id, {"msg": "N/A", "color": "white", "jours": "Pas de données"})
            st.markdown(f"### Pompe {p_id}")
            st.subheader(f":{s['color']}[{s['msg']}]")
            st.write(f"**Echéance :** {s['jours']}")

    st.divider()

    # --- 1. GRAPHIQUE RATIO (CUVE) ---
    st.subheader(f"📊 Ratio de Réconciliation : Cuve {c_id}")
    fig_ratio = go.Figure()
    colors = np.where(df_c['Ratio_Brut'].abs() <= SEUIL_LEGAL, '#00ff88', '#ff4b4b')
    fig_ratio.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c['Ratio_Brut'], mode='lines+markers', marker=dict(color=colors, size=5), name="Ratio Brut"))
    fig_ratio.add_hline(y=SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_ratio.add_hline(y=-SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_ratio.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_ratio, use_container_width=True)

    # --- 2 & 3. GRAPHIQUES CUSUM (POMPES) ---
    st.subheader(f"📈 Santé Métrologique Individuelle (CUSUM)")
    
    for p_id in LIAISONS[c_id]:
        p_str = str(p_id)
        if f'CUSUM_P{p_str}' in df_c.columns:
            fig_p = go.Figure()
            # Tunnel personnel dynamique
            vol_p_cumule = df_c[p_str].cumsum()
            limite_p_litres = vol_p_cumule * (SEUIL_LEGAL / 100)

            fig_p.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c[f'CUSUM_P{p_str}'], name="Erreur (L)", line=dict(color='#00d4ff', width=3)))
            fig_p.add_trace(go.Scatter(x=df_c['Timestamp'], y=limite_p_litres, name="Limite +", line=dict(color='rgba(255, 75, 75, 0.5)', dash='dot')))
            fig_p.add_trace(go.Scatter(x=df_c['Timestamp'], y=-limite_p_litres, name="Limite -", line=dict(color='rgba(255, 75, 75, 0.5)', dash='dot'), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.05)'))

            # Agrandissement à 500px
            fig_p.update_layout(title=f"Pompe {p_id} : Dérive Cumulative Personnalisée", template="plotly_dark", height=500)
            st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("📄 Journal détaillé des anomalies & Prévisions"):
        for m in journal: st.write(m)
