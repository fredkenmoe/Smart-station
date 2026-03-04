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

URL_CUVES = transformer_drive_en_direct("https://drive.google.com/file/d/1BxdKjJB7Difw4vfe4OKylMV_b5PAGUgL/view?usp=sharing")
URL_POMPES = transformer_drive_en_direct("https://drive.google.com/file/d/1H19rgLxGU7wL5VRhDNg2h9_WMf8rFk9s/view?usp=sharing")

LIAISONS = {1: [1, 3], 2: [2, 4]}
SEUIL_LEGAL = 0.5
PROJECTION_NB_POINTS = 96 * 10 

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
        st.error(f"Erreur : {e}")
        return None, []

# --- ANALYSE IA ---
def analyser_ia_complet(df, cols_p):
    cusum_vals = {p: 0.0 for p in cols_p}
    for idx, row in df.iterrows():
        ecart = row['Baisse_Cuve'] - row['Ventes_Totales']
        for p in cols_p:
            ratio = row[p] / row['Ventes_Totales'] if row['Ventes_Totales'] > 0 else 0
            cusum_vals[p] += (ecart * ratio)
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
                y_val = subset[f'CUSUM_P{p_str}'].values
                vol_cum_reel = subset[p_str].sum()
                limite_actuelle = vol_cum_reel * (SEUIL_LEGAL / 100)

                if len(y_val) > 48:
                    # 1. DETECTION DE COURBURE (COMPARAISON DE PENTES)
                    # Pente Fond (48h)
                    model_fond = TheilSenRegressor(random_state=42)
                    n_fond = min(len(y_val), 192)
                    model_fond.fit(np.arange(n_fond).reshape(-1, 1), y_val[-n_fond:])
                    pente_fond = model_fond.coef_[0]

                    # Pente Alerte (12h - Réactivité)
                    model_alerte = TheilSenRegressor(random_state=42)
                    n_alerte = min(len(y_val), 48)
                    model_alerte.fit(np.arange(n_alerte).reshape(-1, 1), y_val[-n_alerte:])
                    pente_alerte = model_alerte.coef_[0]

                    # On prend la pente la plus forte (inclinaison)
                    pente_finale = pente_alerte if abs(pente_alerte) > abs(pente_fond) else pente_fond

                    # 2. PROJECTION DE LA LIMITE DYNAMIQUE
                    debit_moyen_15m = subset[p_str].tail(96).mean()
                    vitesse_limite = debit_moyen_15m * (SEUIL_LEGAL / 100)

                    # 3. CALCUL DE L'INTERSECTION
                    vitesse_rapprochement = abs(pente_finale) - vitesse_limite
                    if vitesse_rapprochement > 0:
                        dist_restante = limite_actuelle - abs(y_val[-1])
                        slots = dist_restante / vitesse_rapprochement
                        j = round((max(0, slots) * 15) / 1440, 1)
                        jours_txt = f"{j} jours" if j > 0 else "DÉPASSÉ"
                    else:
                        jours_txt = "Stable > 1 an"

                    # 4. PREPARATION GRAPHIQUE
                    idx_futur = np.arange(PROJECTION_NB_POINTS)
                    proj_y = y_val[-1] + (idx_futur * pente_finale)
                    proj_lim_sup = limite_actuelle + (idx_futur * vitesse_limite)
                    proj_lim_inf = -limite_actuelle - (idx_futur * vitesse_limite)
                    graph_times = [subset['Timestamp'].max() + pd.Timedelta(minutes=15*i) for i in range(1, PROJECTION_NB_POINTS + 1)]

                    diagnostics_preventifs[p_id] = {
                        "msg": "🔴 CRITIQUE" if "jours" in jours_txt and float(jours_txt.split()[0]) < 7 else "✅ SAIN",
                        "jours": jours_txt, "proj_x": graph_times, "proj_y": proj_y,
                        "lim_sup": proj_lim_sup, "lim_inf": proj_lim_inf
                    }

        # RAPPORT
        anomalies = subset[(subset['Anomaly_Score'] == -1) & (abs(subset['Ratio_Brut']) > SEUIL_LEGAL)]
        for _, row in anomalies.iterrows():
            type_a = "FUITE/VOL" if row['Ratio_Brut'] > 0 else "METROLOGIE"
            echeance = diagnostics_preventifs.get(pompes_cuve[0], {}).get('jours', 'N/A')
            rapport_diagnostic.append(f"🚨 CUVE {id_c} | {row['Timestamp'].strftime('%d/%m %H:%M')} : {type_a} ({row['Ratio_Brut']:.2f}%) - Maintenance : {echeance}")

    return df, rapport_diagnostic, diagnostics_preventifs

# --- AFFICHAGE ---
data, p_ids = charger_donnees(URL_CUVES, URL_POMPES)
if data is not None:
    data, journal, stats = analyser_ia_complet(data, p_ids)
    
    # Sidebar
    st.sidebar.header("📋 État de la Station")
    for p in sorted(p_ids):
        s = stats.get(int(p), {"jours": "N/A", "msg": "N/A"})
        st.sidebar.write(f"**Pompe {p}** : {s['jours']}")
    
    c_id = st.sidebar.selectbox("Sélection Cuve", [1, 2])
    df_c = data[data['ID_Cuve'] == c_id]

    # Metrics
    cols = st.columns(len(LIAISONS[c_id]))
    for i, p_id in enumerate(LIAISONS[c_id]):
        with cols[i]:
            info = stats.get(p_id, {"msg": "N/A", "jours": "N/A"})
            st.metric(f"Pompe {p_id}", info['msg'], info['jours'], delta_color="inverse")

    # Graphes CUSUM
    st.subheader("📈 Détection d'inclinaison et Intersection Limite")
    for p_id in LIAISONS[c_id]:
        p_str = str(p_id)
        if f'CUSUM_P{p_str}' in df_c.columns:
            s_p = stats.get(p_id, {})
            fig = go.Figure()
            lim = df_c[p_str].cumsum() * (SEUIL_LEGAL / 100)
            
            # Réel
            fig.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c[f'CUSUM_P{p_str}'], name="CUSUM", line=dict(color='#00d4ff')))
            fig.add_trace(go.Scatter(x=df_c['Timestamp'], y=lim, name="Limite", line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=df_c['Timestamp'], y=-lim, showlegend=False, line=dict(color='red', dash='dot')))

            # Projection
            if "proj_x" in s_p:
                fig.add_trace(go.Scatter(x=s_p["proj_x"], y=s_p["proj_y"], name="Projection IA", line=dict(color='yellow', dash='dash')))
                fig.add_trace(go.Scatter(x=s_p["proj_x"], y=s_p["lim_sup"], name="Limite Future", line=dict(color='rgba(255,0,0,0.2)', dash='dash')))
                fig.add_trace(go.Scatter(x=s_p["proj_x"], y=s_p["lim_inf"], showlegend=False, line=dict(color='rgba(255,0,0,0.2)', dash='dash')))

            fig.update_layout(template="plotly_dark", height=450, title=f"Pompe {p_id}")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Journal détaillé des anomalies"):
        for m in journal: st.write(m)
