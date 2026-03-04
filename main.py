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
ALPHA = 1.0  # Mis à 1.0 pour une réactivité maximale sans lissage
PROJECTION_NB_POINTS = 96 * 10 # Projection sur 10 jours

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
        df['Ratio_Lisse'] = df.groupby('ID_Cuve')['Ratio_Brut'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        return df, cols_p
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, []

# --- ANALYSE IA ---
def analyser_ia_complet(df, cols_p):
    cusum_vals = {p: 0.0 for p in cols_p}
    for idx, row in df.iterrows():
        ecart_cuve_litres = row['Baisse_Cuve'] - row['Ventes_Totales']
        for p in cols_p:
            ratio_debit = row[p] / row['Ventes_Totales'] if row['Ventes_Totales'] > 0 else 0
            cusum_vals[p] = (cusum_vals[p]) + (ecart_cuve_litres * ratio_debit)
            df.at[idx, f'CUSUM_P{p}'] = cusum_vals[p]

    model_if = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly_Score'] = model_if.fit_predict(df[['Ratio_Lisse']])
    
    rapport_diagnostic = []
    diagnostics_preventifs = {}

    for id_c, pompes_cuve in LIAISONS.items():
        subset = df[df['ID_Cuve'] == id_c].sort_values('Timestamp')
        if subset.empty: continue

        # 1. PROJECTION LINÉAIRE ROBUSTE (Theil-Sen sur 24h)
        for p_id in pompes_cuve:
            p_str = str(p_id)
            if f'CUSUM_P{p_str}' in subset.columns:
                y_val = subset[f'CUSUM_P{p_str}'].values
                vol_p_total = subset[p_str].sum()
                limite_p_litres = vol_p_total * (SEUIL_LEGAL / 100)

                if len(y_val) > 10:
                    model_ts = TheilSenRegressor(random_state=42)
                    n_focus = min(len(y_val), 96) # Focus dernières 24h
                    x_train = np.arange(len(y_val) - n_focus, len(y_val)).reshape(-1, 1)
                    y_train = y_val[-n_focus:]
                    model_ts.fit(x_train, y_train)
                    
                    pente = model_ts.coef_[0]
                    
                    # Projection 10j pour le graphique
                    graph_indices = np.arange(len(y_val), len(y_val) + PROJECTION_NB_POINTS)
                    graph_y = y_val[-1] + (graph_indices - (len(y_val)-1)) * pente
                    graph_times = [subset['Timestamp'].max() + pd.Timedelta(minutes=15*i) for i in range(1, PROJECTION_NB_POINTS + 1)]
                    
                    # Calcul jours restants
                    if (pente > 0 and y_val[-1] < limite_p_litres) or (pente < 0 and y_val[-1] > -limite_p_litres):
                        target = limite_p_litres if pente > 0 else -limite_p_litres
                        nb_slots = (target - y_val[-1]) / pente
                        j = round((nb_slots * 15) / 1440, 1)
                        jours_txt = f"{j} jours" if 0 < j < 365 else "> 1 an"
                    else:
                        jours_txt = "Stable"
                    
                    status = "✅ SAIN"
                    color = "#00d4ff"
                    if abs(y_val[-1]) > limite_p_litres:
                        status, color, jours_txt = "🔴 HORS-NORME", "red", "DÉPASSEMENT"
                    elif "jours" in jours_txt and float(jours_txt.split()[0]) < 7:
                        status, color = "🟡 CRITIQUE", "orange"

                    diagnostics_preventifs[p_id] = {
                        "msg": status, "color": color, "jours": jours_txt, 
                        "graph_y": graph_y, "graph_x": graph_times
                    }

        # 2. RAPPORT DES ANOMALIES (Incluant la Cuve)
        anomalies_brusques = subset[(subset['Anomaly_Score'] == -1) & (abs(subset['Ratio_Brut']) > SEUIL_LEGAL)]
        for _, row_a in anomalies_brusques.iterrows():
            type_ano = "VOL / FUITE" if row_a['Ratio_Brut'] > 0.5 else "MÉTROLOGIE / AIR"
            p_ref = pompes_cuve[0]
            echeance = diagnostics_preventifs.get(p_ref, {}).get('jours', 'N/A')
            rapport_diagnostic.append(f"🚨 CUVE {id_c} | {row_a['Timestamp'].strftime('%d/%m %H:%M')} : {type_ano} ({row_a['Ratio_Brut']:.2f}%) - Échéance maintenance : {echeance}")

    return df, rapport_diagnostic, diagnostics_preventifs

# --- DASHBOARD ---
data, p_ids = charger_donnees(URL_CUVES, URL_POMPES)

if data is not None:
    data, journal, stats = analyser_ia_complet(data, p_ids)
    
    # Barre Latérale : Récapitulatif de toutes les pompes
    st.sidebar.header("📋 Prévisions Maintenance")
    for p in sorted(p_ids):
        s_p = stats.get(int(p), {"jours": "N/A", "msg": "N/A"})
        st.sidebar.write(f"**Pompe {p}** : {s_p['msg']}")
        st.sidebar.caption(f"Échéance : {s_p['jours']}")
        st.sidebar.divider()

    c_id = st.sidebar.selectbox("Sélection Cuve", [1, 2])
    df_c = data[data['ID_Cuve'] == c_id].sort_values('Timestamp')

    cols_h = st.columns(len(LIAISONS[c_id]))
    for i, p_id in enumerate(LIAISONS[c_id]):
        with cols_h[i]:
            s = stats.get(p_id, {"msg": "N/A", "color": "white", "jours": "N/A"})
            st.metric(label=f"Pompe {p_id}", value=s['msg'], delta=s['jours'], delta_color="inverse")

    st.divider()

    # Graph Ratio
    st.subheader(f"📊 Analyse Instantanée (Brut) : Cuve {c_id}")
    fig_ratio = go.Figure()
    colors = np.where(df_c['Ratio_Brut'].abs() <= SEUIL_LEGAL, '#00ff88', '#ff4b4b')
    fig_ratio.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c['Ratio_Brut'], mode='markers', marker=dict(color=colors, size=4), name="Ratio"))
    fig_ratio.add_hline(y=SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_ratio.add_hline(y=-SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_ratio.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_ratio, use_container_width=True)

    # Graph CUSUM avec Pointillés
    st.subheader("📈 Santé Métrologique & Projection 10j (Pointillés)")
    for p_id in LIAISONS[c_id]:
        p_str = str(p_id)
        if f'CUSUM_P{p_str}' in df_c.columns:
            s_p = stats.get(p_id, {})
            fig_p = go.Figure()
            vol_cum = df_c[p_str].cumsum()
            limite = vol_cum * (SEUIL_LEGAL / 100)
            
            # Historique
            fig_p.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c[f'CUSUM_P{p_str}'], name="CUSUM Réel", line=dict(color='#00d4ff', width=3)))
            
            # Projection Linéaire (Pointillés)
            if "graph_x" in s_p:
                fig_p.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["graph_y"], name="Tendance IA (10j)", line=dict(color='yellow', width=2, dash='dot')))
            
            # Tunnel
            fig_p.add_trace(go.Scatter(x=df_c['Timestamp'], y=limite, name="Limite +", line=dict(color='rgba(255, 75, 75, 0.5)', dash='dot')))
            fig_p.add_trace(go.Scatter(x=df_c['Timestamp'], y=-limite, name="Limite -", line=dict(color='rgba(255, 75, 75, 0.5)', dash='dot'), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.05)'))
            
            fig_p.update_layout(template="plotly_dark", height=500, title=f"Pompe {p_id} : Analyse de dérive")
            st.plotly_chart(fig_p, use_container_width=True)

    with st.expander("📄 Journal des anomalies"):
        for m in journal: st.write(m)
