import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import TheilSenRegressor
import base64

# --- CONFIGURATION INTERFACE ---
st.set_page_config(page_title="Smart-Station IA", layout="wide")

# --- FONCTION DE CONVERSION AUTOMATIQUE ONEDRIVE ---
def convertir_onedrive_en_direct(onedrive_url):
    """Transforme un lien de partage OneDrive standard en lien de téléchargement direct."""
    try:
        # Encodage du lien en base64 pour l'API OneDrive
        data_bytes64 = base64.b64encode(bytes(onedrive_url, 'utf-8'))
        data_bytes64_string = data_bytes64.decode('utf-8').replace('/', '_').replace('+', '-').rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_string}/root/content"
    except Exception:
        return onedrive_url

# --- PARAMÈTRES ET LIENS ---
# COLLE TES LIENS DE PARTAGE ONEDRIVE ICI
URL_CUVES_SHARE = "https://energenic-my.sharepoint.com/:x:/r/personal/anicet_bibout_energenic_cloud/Documents/Documents/SMARTSTATION/ALGO/Simulateur/Inventaire_Cuves_Variation.csv?d=w62aa55394c1a40ceb273281ab619d0f2&csf=1&web=1&e=GB3foj"
URL_POMPES_SHARE = "https://energenic-my.sharepoint.com/:x:/r/personal/anicet_bibout_energenic_cloud/Documents/Documents/SMARTSTATION/ALGO/Simulateur/Transactions_Cuves_Variation.csv?d=wad35079cad5d415ea0de5f3f4e2ca1d4&csf=1&web=1&e=2KIycc"

# Conversion automatique
URL_CUVES = convertir_onedrive_en_direct(URL_CUVES_SHARE)
URL_POMPES = convertir_onedrive_en_direct(URL_POMPES_SHARE)

LIAISONS = {1: [1, 3], 2: [2, 4]}
SEUIL_LEGAL = 0.5
ALPHA = 0.99

# --- LOGIQUE DE RÉCONCILIATION ---
@st.cache_data
def charger_et_reconcilier(url_c, url_p):
    try:
        df_cuves = pd.read_csv(url_c)
        df_pompes = pd.read_csv(url_p)
        
        # Formatage des dates
        df_cuves['Timestamp'] = pd.to_datetime(df_cuves['Date'] + ' ' + df_cuves['Heure'], dayfirst=True)
        df_pompes['Timestamp'] = pd.to_datetime(df_pompes['Date'] + ' ' + df_pompes['Heure'], dayfirst=True)
        df_pompes['Slot'] = df_pompes['Timestamp'].dt.floor('15min') + pd.Timedelta(minutes=15)

        # Pivot des ventes pompes
        p_pivot = df_pompes.pivot_table(index=['ID_Cuve', 'Slot'], columns='ID_Pompe', 
                                        values='Volume_Vendu', aggfunc='sum').fillna(0).reset_index()

        # Calcul de la baisse réelle de la cuve
        df_cuves = df_cuves.sort_values(['ID_Cuve', 'Timestamp'])
        df_cuves['Baisse_Cuve'] = df_cuves.groupby('ID_Cuve')['Volume_L'].shift(1) - df_cuves['Volume_L']

        # Fusion des données
        df = pd.merge(df_cuves, p_pivot, left_on=['ID_Cuve', 'Timestamp'], right_on=['ID_Cuve', 'Slot'], how='inner')
        
        # Identification des colonnes de pompes
        cols_p = [c for c in df.columns if str(c).isdigit()]
        df['Ventes_Totales'] = df[cols_p].sum(axis=1)
        
        # Nettoyage et calcul des ratios
        df = df[(df['Baisse_Cuve'] >= 0) & (df['Ventes_Totales'] > 0)].copy()
        df['Ratio_Brut'] = ((df['Baisse_Cuve'] - df['Ventes_Totales']) / df['Ventes_Totales']) * 100
        df['Ratio_Lisse'] = df.groupby('ID_Cuve')['Ratio_Brut'].transform(lambda x: x.rolling(window=8, min_periods=1).mean())
        
        return df, cols_p
    except Exception as e:
        st.error(f"Erreur de lecture des données : {e}")
        return None, []

# --- LOGIQUE IA ET DIAGNOSTIC ---
def calculer_ia_complet(df, cols_p):
    pompes_ids = [int(c) for c in cols_p]
    cusum_vals = {p: 0.0 for p in pompes_ids}
    
    # 1. Calcul CUSUM (Accumulation de l'erreur en Litres)
    for idx, row in df.iterrows():
        ecart_L = row['Baisse_Cuve'] - row['Ventes_Totales']
        for p in pompes_ids:
            ratio = row[str(p)] / row['Ventes_Totales'] if row['Ventes_Totales'] > 0 else 0
            # Mise à jour avec facteur d'oubli (Alpha)
            cusum_vals[p] = (ALPHA * cusum_vals[p]) + (ecart_L * ratio)
            df.at[idx, f'CUSUM_P{p}'] = cusum_vals[p]

    # 2. Isolation Forest (Détection des pics anormaux)
    model_if = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly_Score'] = model_if.fit_predict(df[['Ratio_Lisse']])
    
    rapport = []
    projections = {}

    for id_c, pompes_cuve in LIAISONS.items():
        subset = df[df['ID_Cuve'] == id_c].sort_values('Timestamp')
        
        # Analyse des points critiques
        anomalies = subset[(subset['Anomaly_Score'] == -1) & (abs(subset['Ratio_Brut']) > SEUIL_LEGAL)]
        for _, r in anomalies.iterrows():
            if r['Ratio_Brut'] > 0.5:
                type_ano = "🚨 VOL / FUITE POSSIBLE"
            elif r['Ratio_Brut'] < -0.5:
                type_ano = "🌬️ MÉTROLOGIE / PRÉSENCE D'AIR"
            else:
                type_ano = "⚠️ ANOMALIE NON CLASSÉE"
            rapport.append(f"{type_ano} ({r['Ratio_Brut']:.2f}%) détecté le {r['Timestamp'].strftime('%d/%m %H:%M')}")

        # 3. Régression Theil-Sen (Modèle de tendance robuste)
        for p_id in pompes_cuve:
            y = subset[f'CUSUM_P{p_id}'].values
            x = np.arange(len(y)).reshape(-1, 1)
            if len(y) > 20:
                model_ts = TheilSenRegressor(random_state=42)
                model_ts.fit(x, y)
                # Projection sur les 100 prochains créneaux
                future_x = np.arange(len(y), len(y) + 100).reshape(-1, 1)
                future_y = model_ts.predict(future_x)
                projections[p_id] = (future_x, future_y)
                
    return df, rapport, projections

# --- INTERFACE UTILISATEUR ---
st.title("⛽ Smart-Station : Monitoring IA & Réconciliation")

df_final, pompes_detectees = charger_et_reconcilier(URL_CUVES, URL_POMPES)

if df_final is not None:
    df_final, journal, proj_dict = calculer_ia_complet(df_final, pompes_detectees)
    
    cuve_id = st.sidebar.selectbox("Sélectionner la Cuve", [1, 2])
    df_cuve = df_final[df_final['ID_Cuve'] == cuve_id]

    # --- GRAPHIQUE 1 : ÉCARTS BRUTS ---
    st.subheader(f"Analyse des Écarts de Réconciliation - Cuve {cuve_id}")
    fig_ratio = go.Figure()
    # Code couleur : Vert si conforme, Rouge si hors-norme
    colors = np.where(df_cuve['Ratio_Brut'].abs() <= SEUIL_LEGAL, '#00ff88', '#ff4b4b')
    
    fig_ratio.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=df_cuve['Ratio_Brut'], 
                                   mode='markers', marker=dict(color=colors, size=6), name="Écart Brut %"))
    
    # Lignes de seuils légaux
    fig_ratio.add_hline(y=SEUIL_LEGAL, line_dash="dash", line_color="red", annotation_text="Limite Haute (+0.5%)")
    fig_ratio.add_hline(y=-SEUIL_LEGAL, line_dash="dash", line_color="red", annotation_text="Limite Basse (-0.5%)")
    
    fig_ratio.update_layout(template="plotly_dark", height=400, yaxis_title="Erreur (%)")
    st.plotly_chart(fig_ratio, use_container_width=True)

    # --- GRAPHIQUE 2 : SANTÉ MÉTROLOGIQUE (CUSUM) ---
    st.subheader("Diagnostic Métrologique par Pompe & Prédictions IA")
    cols = st.columns(2)
    
    for i, p_id in enumerate(LIAISONS[cuve_id]):
        with cols[i]:
            fig_p = go.Figure()
            # Calcul du tunnel de tolérance cumulé
            vol_cum = df_cuve[str(p_id)].cumsum()
            limite = vol_cum * (SEUIL_LEGAL / 100)
            
            # Tracé de l'erreur cumulée réelle
            fig_p.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=df_cuve[f'CUSUM_P{p_id}'], 
                                       name="Erreur Cumulée (L)", line=dict(color='#00d4ff', width=2)))
            
            # Tracé du tunnel de tolérance
            fig_p.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=limite, name="Tolérance +", line=dict(color='red', dash='dot')))
            fig_p.add_trace(go.Scatter(x=df_cuve['Timestamp'], y=-limite, name="Tolérance -", line=dict(color='red', dash='dot')))
            
            # Affichage de la ligne de tendance IA (Theil-Sen)
            if p_id in proj_dict:
                fx, fy = proj_dict[p_id]
                last_t = df_cuve['Timestamp'].max()
                future_t = [last_t + pd.Timedelta(minutes=15*j) for j in range(len(fy))]
                fig_p.add_trace(go.Scatter(x=future_t, y=fy, name="Tendance IA (Prédiction)", line=dict(color='yellow', dash='dash')))
            
            fig_p.update_layout(title=f"Pompe {p_id} (Analyse de dérive)", template="plotly_dark", height=350)
            st.plotly_chart(fig_p, use_container_width=True)

    # --- SECTION : JOURNAL DE BORD IA ---
    st.divider()
    with st.expander("📄 Consulter le Rapport de Diagnostic IA (Points Critiques)"):
        if journal:
            for msg in journal:
                st.write(msg)
        else:
            st.success("Aucune anomalie critique détectée sur la période.")
