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
PROJECTION_GRAPHE_POINTS = 96 * 10 # 10 jours pour le visuel
PROJECTION_ANALYSE_POINTS = 96 * 365 # 1 an pour le calcul d'échéance

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
                vol_p_total = subset[p_str].sum()
                lim_actuelle = vol_p_total * (SEUIL_LEGAL / 100)

                if len(y_val) > 192:
                    # ANALYSE 5 INTERVALLES (10 JOURS) POUR COURBURE
                    pentes = []
                    for i in range(5):
                        fin = len(y_val) - (i * 192)
                        debut = max(0, fin - 192)
                        model_ts = TheilSenRegressor(random_state=42).fit(np.arange(fin-debut).reshape(-1, 1), y_val[debut:fin])
                        pentes.insert(0, model_ts.coef_[0])

                    v_actuelle = pentes[-1]
                    accel = (pentes[-1] - pentes[0]) / (4 * 192)
                    
                    # PROJECTION VISUELLE (10 JOURS)
                    t_visu = np.arange(1, PROJECTION_GRAPHE_POINTS + 1)
                    graph_y = y_val[-1] + (v_actuelle * t_visu) + (0.5 * accel * (t_visu**2))
                    graph_times = [subset['Timestamp'].max() + pd.Timedelta(minutes=15*i) for i in range(1, PROJECTION_GRAPHE_POINTS + 1)]
                    
                    # PROJECTION ANALYSE (1 AN) POUR CALCUL ÉCHÉANCE
                    t_an = np.arange(1, PROJECTION_ANALYSE_POINTS + 1)
                    y_an = y_val[-1] + (v_actuelle * t_an) + (0.5 * accel * (t_an**2))
                    
                    debit_m = subset[p_str].tail(96).mean()
                    v_lim = debit_m * (SEUIL_LEGAL / 100)
                    lim_futures = lim_actuelle + (t_an * v_lim)
                    
                    # TROUVER INTERSECTION SUR 1 AN
                    jours_txt = "> 1 an"
                    collision = np.where(np.abs(y_an) >= lim_futures)[0]
                    if len(collision) > 0:
                        j = round((collision[0] * 15) / 1440, 1)
                        jours_txt = f"{j} jours" if j > 0 else "DÉPASSÉ"

                    # DONNÉES GRAPHIQUE LIMITES (10j)
                    lim_sup_visu = lim_actuelle + (t_visu * v_lim)
                    lim_inf_visu = -lim_actuelle - (t_visu * v_lim)

                    diagnostics_preventifs[p_id] = {
                        "msg": "🔴 HORS-NORME" if "DÉP" in jours_txt else ("🟡 CRITIQUE" if "jours" in jours_txt and float(jours_txt.split()[0]) < 7 else "✅ SAIN"),
                        "jours": jours_txt, "graph_x": graph_times, "graph_y": graph_y,
                        "lim_sup": lim_sup_visu, "lim_inf": lim_inf_visu, "accel": round(abs(pentes[-1]/pentes[0]) if pentes[0]!=0 else 1.0, 2)
                    }

    return df, rapport_diagnostic, diagnostics_preventifs

# --- DASHBOARD ---
data, p_ids = charger_donnees(URL_CUVES, URL_POMPES)
if data is not None:
    data, journal, stats = analyser_ia_complet(data, p_ids)
    
    st.sidebar.header("📋 Maintenance")
    c_id = st.sidebar.selectbox("Sélection Cuve", [1, 2])
    df_c = data[data['ID_Cuve'] == c_id].sort_values('Timestamp')

    # GRAPHE RATIO (BRUT)
    st.subheader(f"📊 Ratio Brut : Cuve {c_id}")
    fig_r = go.Figure()
    colors = np.where(df_c['Ratio_Brut'].abs() <= SEUIL_LEGAL, '#00ff88', '#ff4b4b')
    fig_r.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c['Ratio_Brut'], mode='markers', marker=dict(color=colors, size=4)))
    fig_r.add_hline(y=SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_r.add_hline(y=-SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_r.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_r, use_container_width=True)

    # GRAPHE CUSUM (PROJECTION COURBÉE 10J)
    st.subheader("📈 CUSUM : Projection Courbe (Zoom 10j)")
    for p_id in LIAISONS[c_id]:
        p_str = str(p_id)
        if f'CUSUM_P{p_str}' in df_c.columns:
            s_p = stats.get(p_id, {})
            fig = go.Figure()
            lim_h = df_c[p_str].cumsum() * (SEUIL_LEGAL / 100)
            
            fig.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c[f'CUSUM_P{p_str}'], name="Réel", line=dict(color='#00d4ff', width=3)))
            fig.add_trace(go.Scatter(x=df_c['Timestamp'], y=lim_h, name="Limites", line=dict(color='rgba(255,0,0,0.3)', dash='dot')))
            fig.add_trace(go.Scatter(x=df_c['Timestamp'], y=-lim_h, showlegend=False, line=dict(color='rgba(255,0,0,0.3)', dash='dot')))

            if "graph_x" in s_p:
                fig.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["graph_y"], name="Proj. Courbe", line=dict(color='yellow', dash='dot')))
                fig.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["lim_sup"], name="Lim. Future", line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["lim_inf"], showlegend=False, line=dict(color='red', dash='dash')))

            fig.update_layout(template="plotly_dark", height=450, title=f"Pompe {p_id} | Échéance : {s_p.get('jours')} | Accel : x{s_p.get('accel')}")
            st.plotly_chart(fig, use_container_width=True)
