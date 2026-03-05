import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import TheilSenRegressor

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart-Station IA Energenic", layout="wide")

st.title("🚀 Smart-Station IA Energenic : Monitoring & Maintenance Prédictive")

# --- CONSTANTES ---
POINTS_PAR_JOUR = 96  # 15 min * 96 = 24h
FENETRE_ETUDE_MAX = POINTS_PAR_JOUR * 60  # Mémoire de 60 jours pour le calcul
POINTS_AFFICHAGE_ZOOM = POINTS_PAR_JOUR * 10  # Focus visuel de 10 jours
PROJECTION_GRAPHE_POINTS = POINTS_PAR_JOUR * 10  # Projection de 10 jours
PROJECTION_ANALYSE_POINTS = POINTS_PAR_JOUR * 365 # Calcul sur 1 an
SEUIL_LEGAL = 0.5

import pandas as pd
import base64

def obtenir_lien_direct(url):
    """
    Transforme dynamiquement un lien de partage en flux de données brut (Raw Data).
    Supporte : Google Drive, OneDrive et SharePoint Energenic.
    """
    try:
        # CAS 1 : GOOGLE DRIVE
        if "drive.google.com" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # CAS 2 : SHAREPOINT / ONEDRIVE (Encodage Base64 pour l'API Microsoft)
        elif "sharepoint.com" in url or "1drv.ms" in url:
            # On génère un ID de partage encodé en Base64 pour l'API Graph
            encoded_url = base64.b64encode(bytes(url, 'utf-8')).decode('utf-8')
            # On nettoie l'encodage pour l'URL (standard Microsoft)
            res_url = "u!" + encoded_url.replace('/', '_').replace('+', '-').rstrip('=')
            return f"https://api.onedrive.com/v1.shares/{res_url}/root/content"
        
        return url # Retourne l'URL tel quel si c'est déjà un lien direct
    except Exception as e:
        st.error(f"Erreur de conversion du lien Cloud : {e}")
        return url

# --- UTILISATION ---

URL_CUVES = obtenir_lien_direct("https://drive.google.com/file/d/1BxdKjJB7Difw4vfe4OKylMV_b5PAGUgL/view?usp=sharing")
URL_POMPES = obtenir_lien_direct("https://drive.google.com/file/d/1H19rgLxGU7wL5VRhDNg2h9_WMf8rFk9s/view?usp=sharing")

LIAISONS = {1: [1, 3], 2: [2, 4]}

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

def analyser_ia_complet(df, cols_p):
    # Calcul CUSUM Global
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
                y_full = subset[f'CUSUM_P{p_str}'].values
                
                # --- LOGIQUE DYNAMIQUE : 60 JOURS SI DISPO ---
                nb_points_etude = min(len(y_full), FENETRE_ETUDE_MAX)
                y_etude = y_full[-nb_points_etude:]
                
                vol_p_total = subset[p_str].sum()
                lim_actuelle = vol_p_total * (SEUIL_LEGAL / 100)

                if len(y_etude) > 192:
                    pentes = []
                    # On divise la fenêtre disponible en 5 blocs
                    taille_bloc = len(y_etude) // 5
                    for i in range(5):
                        fin = len(y_etude) - (i * taille_bloc)
                        debut = max(0, fin - taille_bloc)
                        model_ts = TheilSenRegressor(random_state=42).fit(np.arange(fin-debut).reshape(-1, 1), y_etude[debut:fin])
                        pentes.insert(0, model_ts.coef_[0])

                    v_actuelle = pentes[-1]
                    accel = (pentes[-1] - pentes[0]) / (len(y_etude)) if len(y_etude)>0 else 0
                    
                    t_visu = np.arange(1, PROJECTION_GRAPHE_POINTS + 1)
                    graph_y = y_etude[-1] + (v_actuelle * t_visu) + (0.5 * accel * (t_visu**2))
                    graph_times = [subset['Timestamp'].max() + pd.Timedelta(minutes=15*i) for i in range(1, PROJECTION_GRAPHE_POINTS + 1)]
                    
                    t_an = np.arange(1, PROJECTION_ANALYSE_POINTS + 1)
                    y_an = y_etude[-1] + (v_actuelle * t_an) + (0.5 * accel * (t_an**2))
                    v_lim = subset[p_str].tail(96).mean() * (SEUIL_LEGAL / 100)
                    lim_futures = lim_actuelle + (t_an * v_lim)
                    
                    jours_txt = "> 1 an"
                    collision = np.where(np.abs(y_an) >= lim_futures)[0]
                    if len(collision) > 0:
                        j = round((collision[0] * 15) / 1440, 1)
                        jours_txt = f"{j} jours" if j > 0 else "DÉPASSÉ"

                    diagnostics_preventifs[p_id] = {
                        "msg": "🔴 HORS-NORME" if "DÉP" in jours_txt else ("🟡 CRITIQUE" if "jours" in jours_txt and float(jours_txt.split()[0]) < 7 else "✅ SAIN"),
                        "jours": jours_txt, "graph_x": graph_times, "graph_y": graph_y,
                        "lim_sup": lim_actuelle + (t_visu * v_lim), "lim_inf": -lim_actuelle - (t_visu * v_lim),
                        "accel": round(abs(pentes[-1]/pentes[0]) if pentes[0]!=0 else 1.0, 2)
                    }

        # Rapport Anomalies (basé sur l'historique de la cuve)
       # --- RAPPORT IA & ANOMALIES ---
        anomalies_brusques = subset[(subset['Anomaly_Score'] == -1) & (abs(subset['Ratio_Brut']) > SEUIL_LEGAL)]
        for _, row_a in anomalies_brusques.iterrows():
            if row_a['Ratio_Brut'] > 0.5: type_ano = "VOL / FUITE"
            elif row_a['Ratio_Brut'] < -0.5: type_ano = "MÉTROLOGIE / PRÉSENCE D'AIR"
            else: type_ano = "ANOMALIE NON CLASSIFIÉE"

            p_ref = pompes_cuve[0]
            echeance = diagnostics_preventifs.get(p_ref, {}).get('jours', 'N/A')
            rapport_diagnostic.append(
                f"🚨 {row_a['Timestamp'].strftime('%d/%m %H:%M')} | Cuve {id_c} : {type_ano} ({row_a['Ratio_Brut']:.2f}%) - Prévision Maintenance : {echeance}"
            )

    return df, rapport_diagnostic, diagnostics_preventifs

# --- DASHBOARD ---
data, p_ids = charger_donnees(URL_CUVES, URL_POMPES)
if data is not None:
    data, journal, stats = analyser_ia_complet(data, p_ids)
    
    # Sidebar
    st.sidebar.header("📋 Maintenance Prédictive")
    for p in sorted(p_ids):
        s_p = stats.get(int(p), {"jours": "N/A", "msg": "N/A"})
        st.sidebar.write(f"**Pompe {p}** : {s_p['msg']}")
        st.sidebar.caption(f"Échéance : {s_p['jours']}")
        st.sidebar.divider()

    c_id = st.sidebar.selectbox("Sélection Cuve", [1, 2])
    df_c = data[data['ID_Cuve'] == c_id].sort_values('Timestamp')

    # 📊 GRAPHE RATIO (HISTORIQUE COMPLET)
    st.subheader(f"📊 Ratio de Réconciliation : Cuve {c_id} (Historique Complet)")
    fig_ratio = go.Figure()
    colors = np.where(df_c['Ratio_Brut'].abs() <= SEUIL_LEGAL, '#00ff88', '#ff4b4b')
    fig_ratio.add_trace(go.Scatter(x=df_c['Timestamp'], y=df_c['Ratio_Brut'], mode='markers', marker=dict(color=colors, size=4), name="Ratio"))
    fig_ratio.add_hline(y=SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_ratio.add_hline(y=-SEUIL_LEGAL, line_dash="dash", line_color="#ff4b4b")
    fig_ratio.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_ratio, use_container_width=True)

    # 📈 GRAPHE CUSUM (ZOOM 10J PASSÉ + 10J PROJETÉ)
    st.subheader(f"📈Santé Métrologique")
    for p_id in LIAISONS[c_id]:
        p_str = str(p_id)
        if f'CUSUM_P{p_str}' in df_c.columns:
            s_p = stats.get(p_id, {})
            
            # --- FOCALISATION VISUELLE (10 DERNIERS JOURS) ---
            df_zoom = df_c.tail(POINTS_AFFICHAGE_ZOOM)
            
            fig = go.Figure()
            lim_h = df_zoom[p_str].cumsum() * (SEUIL_LEGAL / 100)
            
            # Passé Focalisé
            fig.add_trace(go.Scatter(x=df_zoom['Timestamp'], y=lim_h, name="Limite (+)", line=dict(color='rgba(255, 75, 75, 0.4)', dash='dash')))
            fig.add_trace(go.Scatter(x=df_zoom['Timestamp'], y=-lim_h, showlegend=False, line=dict(color='rgba(255, 75, 75, 0.4)', dash='dash')))
            fig.add_trace(go.Scatter(x=df_zoom['Timestamp'], y=df_zoom[f'CUSUM_P{p_str}'], name="CUSUM Réel (Zoom)", line=dict(color='#00d4ff', width=4)))
            
            # Futur Projeté
            if "graph_x" in s_p:
                fig.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["graph_y"], name="Proj. IA (10j)", line=dict(color='yellow', width=3, dash='dot')))
                fig.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["lim_sup"], name="Lim. Future (+)", line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=s_p["graph_x"], y=s_p["lim_inf"], showlegend=False, line=dict(color='red', dash='dash')))
            
            fig.update_layout(template="plotly_dark", height=450, title=f"Pompe {p_id} | Accélération : x{s_p.get('accel', 1.0)}")
            st.plotly_chart(fig, use_container_width=True)

    # JOURNAL
    st.divider()
    st.subheader("📄 Journal des Anomalies")
    for m in journal[-15:]: # On montre les 15 dernières
        st.write(m)
