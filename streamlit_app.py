import streamlit as st
import pandas as pd
import csv
import chardet
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import optionnel de reportlab (PDF)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

st.set_page_config(page_title="📊 Analyseur Marketing", layout="wide")

# CSS pour améliorer le rendu
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Analyseur de Données Marketing</p>', unsafe_allow_html=True)
st.write("**Importez vos fichiers CSV/XLS/XLSX — Visualisez instantanément vos données**")

# ========================================
# SECTION 1 : IMPORT MULTI-FICHIERS
# ========================================

st.sidebar.title("📁 Import de fichiers")
uploaded_files = st.sidebar.file_uploader(
    "Glissez vos fichiers ici (CSV, XLSX ou XLS)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Vous pouvez importer plusieurs fichiers en même temps"
)

@st.cache_data
def load_single_file(file_content, file_name):
    """Charge un fichier CSV/Excel de manière robuste.
    - Gère 'sep=,' en première ligne.
    - Ignore les lignes de titre/metadata avant l’en-tête.
    - Supporte plusieurs tableaux dans un même CSV.
    - Supporte les formats Excel .xls et .xlsx.
    Retourne un dictionnaire {nom_feuille: dataframe} ou {nom_fichier: dataframe}.
    """
    try:
        lower_name = file_name.lower()
        if lower_name.endswith('.csv'):
            # Détection encodage et normalisation des fins de ligne
            result = chardet.detect(file_content)
            encoding = result.get('encoding') or 'utf-8'
            text = file_content.decode(encoding, errors='ignore').replace('\r\n', '\n').replace('\r', '\n')
            lines = text.split('\n')

            # 1) Détecter 'sep=' et le sauter
            sep = None
            start_idx = 0
            if lines and lines[0].strip().lower().startswith('sep='):
                cand = lines[0].split('=', 1)[1].strip()
                sep = cand[0] if cand else ','
                start_idx = 1

            # 2) Détecter le séparateur si non fixé
            if sep is None:
                sample = '\n'.join(lines[:50])
                try:
                    sep = csv.Sniffer().sniff(sample).delimiter
                except Exception:
                    sep = ',' if sample.count(',') >= sample.count(';') else ';'

            # 3) Trouver la vraie ligne d'en-tête (ignore titres/metadata comme "Vues")
            header_idx = start_idx
            for i in range(start_idx, min(len(lines), 200)):
                raw = lines[i].lstrip('\ufeff').strip()
                if not raw or raw.lower().startswith('sep='):
                    continue
                # en-tête plausible si contient au moins un séparateur
                if sep in raw and raw.count(sep) >= 1:
                    header_idx = i
                    break

            # 4) Contenu utile
            clean_text = '\n'.join(lines[header_idx:]).lstrip('\ufeff').strip()

            # 5) Essayer de lire plusieurs blocs (plusieurs tableaux séparés par lignes vides)
            blocks = [b.strip() for b in clean_text.split('\n\n') if b.strip()]
            frames = []
            table_counter = 0
            for b in blocks:
                # Détection d'en-têtes internes
                sub_lines_all = [ln for ln in b.split('\n') if ln is not None]
                sub_lines = [ln for ln in sub_lines_all if ln.strip() != '']
                if not sub_lines:
                    continue
                base_header = sub_lines[0].strip()

                # Trouver positions d'en-tête candidates
                header_positions = []
                for j, ln in enumerate(sub_lines):
                    s = ln.strip()
                    if not s:
                        continue
                    if s == base_header:
                        header_positions.append(j)
                        continue
                    if sep in s:
                        fields = [x.strip().strip('"') for x in s.split(sep)]
                        if len(fields) >= 1:
                            non_digit_text = 0
                            for f in fields:
                                f2 = f.replace('\xa0', '').strip()
                                has_alpha = any(ch.isalpha() for ch in f2)
                                has_digit = any(ch.isdigit() for ch in f2)
                                if has_alpha and not has_digit:
                                    non_digit_text += 1
                            if non_digit_text >= max(1, len(fields)//2):
                                header_positions.append(j)

                header_positions = sorted(set(header_positions))

                if len(header_positions) > 1:
                    for s_idx, start in enumerate(header_positions):
                        end = header_positions[s_idx + 1] if s_idx + 1 < len(header_positions) else len(sub_lines)
                        seg = '\n'.join(sub_lines[start:end]).strip()
                        if not seg:
                            continue
                        try:
                            df_block = pd.read_csv(StringIO(seg), sep=sep, engine='python', on_bad_lines='skip')
                            if df_block.shape[0] > 0 and df_block.shape[1] >= 1:
                                table_counter += 1
                                df_block['_table_id'] = table_counter
                                frames.append(df_block)
                        except Exception:
                            continue
                else:
                    try:
                        df_block = pd.read_csv(StringIO(b), sep=sep, engine='python', on_bad_lines='skip')
                        if df_block.shape[0] > 0 and df_block.shape[1] >= 1:
                            table_counter += 1
                            df_block['_table_id'] = table_counter
                            frames.append(df_block)
                    except Exception:
                        continue

            if frames:
                df_out = pd.concat(frames, ignore_index=True, sort=False)
            else:
                df_out = pd.read_csv(StringIO(clean_text), sep=sep, engine='python', on_bad_lines='skip')

            # Nettoyage léger des noms de colonnes
            df_out.columns = [str(c).strip() for c in df_out.columns]
            return {file_name: df_out}

        # Excel (.xls / .xlsx)
        def _cleanup_excel_sheet(df):
            """
            Nettoie une feuille Excel en cherchant la vraie ligne d'en-tête.
            On cherche la ligne avec le plus de colonnes non-vides dans les 20 premières lignes.
            """
            if df.empty:
                return df
                
            # Scan des 20 premières lignes pour trouver le header
            max_cols = 0
            header_idx = 0
            
            # On considère aussi la première ligne (les headers actuels) comme potentielle donnée
            # Donc on recharge ou on travaille sur le df brut
            # Ici on travaille sur le df déjà chargé, potentiellement avec des headers faux
            
            # Stratégie : convertir tout en string pour compter
            df_str = df.astype(str)
            
            # Vérifier la "ligne" des headers actuels
            current_header_filled = sum(1 for c in df.columns if not str(c).startswith('Unnamed:'))
            
            candidates = []
            candidates.append((0, current_header_filled)) # Index virtuel -1 ramené à 0
            
            # Vérifier les lignes de données
            for i in range(min(20, len(df))):
                row = df_str.iloc[i]
                # Compter les valeurs non-null par rapport à "nan" ou chaîne vide
                filled = sum(1 for x in row if x.lower() not in ['nan', 'none', '', 'nat'])
                candidates.append((i + 1, filled)) # +1 car on est après le header
            
            # Trouver la meilleure ligne
            best_row, count = max(candidates, key=lambda x: x[1])
            
            # Si c'était la ligne d'en-tête actuelle (0), on garde tel quel
            # Attention: best_row est un index relatif (0 = headers actuels, 1 = 1ere ligne de data...)
            if count <= 1: 
                # Pas assez de colonnes pour être un tableau sérieux -> on garde
                pass
            elif best_row > 0:
                # Promouvoir la ligne 'best_row - 1' (index df) en header
                new_header = df.iloc[best_row - 1]
                df = df.iloc[best_row:].reset_index(drop=True)
                df.columns = new_header
                
            return df

        try:
            # Lire sans header d'abord pour avoir la main ou lire normalement
            # Mieux: Lire normalement et nettoyer ensuite
            excel_dict = pd.read_excel(BytesIO(file_content), sheet_name=None)
            cleaned_dict = {}
            for sheet_name, sheet_df in excel_dict.items():
                cleaned_dict[sheet_name] = _cleanup_excel_sheet(sheet_df)
            return cleaned_dict
        except Exception:
            # Tentatives ciblées par engine
            if lower_name.endswith('.xls'):
                try:
                    excel_dict = pd.read_excel(BytesIO(file_content), engine='xlrd', sheet_name=None)
                    cleaned_dict = {}
                    for sheet_name, sheet_df in excel_dict.items():
                        cleaned_dict[sheet_name] = _cleanup_excel_sheet(sheet_df)
                    return cleaned_dict
                except Exception as e:
                    st.error(f"❌ Impossible de lire le fichier Excel (.xls) : {file_name}. Installez 'xlrd'. Erreur: {e}")
                    return None
            else:
                try:
                    excel_dict = pd.read_excel(BytesIO(file_content), engine='openpyxl', sheet_name=None)
                    cleaned_dict = {}
                    for sheet_name, sheet_df in excel_dict.items():
                        cleaned_dict[sheet_name] = _cleanup_excel_sheet(sheet_df)
                    return cleaned_dict
                except Exception as e:
                    st.error(f"❌ Impossible de lire le fichier Excel : {file_name}. Erreur: {e}")
                    return None

    except Exception as e:
        st.error(f"❌ Erreur {file_name}: {str(e)}")
        return None

# Chargement et fusion des fichiers
all_sheets = {} # Dictionnaire pour stocker toutes les feuilles de tous les fichiers
if uploaded_files:
    with st.spinner('⏳ Chargement des fichiers...'):
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            loaded_data = load_single_file(file_content, uploaded_file.name)
            
            if loaded_data:
                for sheet_name, df_sheet in loaded_data.items():
                    # Créer un nom unique pour chaque feuille : "NomFichier - NomFeuille"
                    unique_name = f"{uploaded_file.name} - {sheet_name}" if len(loaded_data) > 1 else uploaded_file.name
                    df_sheet['_fichier_source'] = uploaded_file.name
                    df_sheet['_feuille_source'] = sheet_name
                    all_sheets[unique_name] = df_sheet

    if all_sheets:
        st.sidebar.success(f"✅ {len(all_sheets)} feuille(s) chargée(s)")
        
        # Affichage des schémas individuels
        st.subheader("📋 Aperçu des feuilles individuelles")
        for name, sheet_df in all_sheets.items():
            with st.expander(f"Feuille : {name}", expanded=False):
                st.write(f"**Dimensions :** {sheet_df.shape[0]} lignes, {sheet_df.shape[1]} colonnes")
                st.dataframe(sheet_df.head(50), use_container_width=True)
                
        # Option de fusion
        st.markdown("---")
        if st.button("🔄 Fusionner toutes les feuilles pour l'analyse globale", type="primary"):
            dfs = list(all_sheets.values())
            df = pd.concat(dfs, ignore_index=True, sort=False)
            st.success(f"✅ Fusion réussie : {len(df):,} lignes au total")
        else:
            df = None # On ne fait pas l'analyse globale tant que pas cliqué
            st.info("👆 Cliquez sur le bouton ci-dessus pour lancer l'analyse complète (fusion des données).")

    else:
        st.sidebar.error("❌ Aucun fichier valide")
        df = None
else:
    df = None

# ========================================
# SECTION 2 : TRAITEMENT DES DONNÉES
# ========================================

# ========================================
# SECTION 2 : FONCTION D'ANALYSE (REUTILISABLE)
# ========================================

def detect_column_types(df):
    """Détecte et convertit automatiquement les types de colonnes."""
    df = df.copy()

    # Conversion automatique des dates
    for col in df.columns:
        if col == '_fichier_source':
            continue
        if df[col].dtype == 'object':
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() / len(df) > 0.5:
                    df[col] = converted
            except:
                pass

    # Conversion automatique des nombres
    for col in df.columns:
        if col == '_fichier_source':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if df[col].dtype == 'object':
            try:
                test_series = df[col].dropna().astype(str).head(100)
                cleaned = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace('\xa0', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .str.replace('€', '', regex=False)
                    .str.replace('$', '', regex=False)
                    .str.replace('£', '', regex=False)
                    .str.replace('%', '', regex=False)
                    .str.replace('−', '-', regex=False)
                    .str.replace('+', '', regex=False)
                )
                converted = pd.to_numeric(cleaned, errors='coerce')
                valid_ratio = converted.notna().sum() / len(df)
                if valid_ratio >= 0.5:
                    df[col] = converted
            except:
                pass

    # Classification des colonnes
    numeric_cols = [c for c in df.columns if c != '_fichier_source' and pd.api.types.is_numeric_dtype(df[c])]
    date_cols = [c for c in df.columns if c != '_fichier_source' and pd.api.types.is_datetime64_any_dtype(df[c])]
    text_cols = [c for c in df.columns if c != '_fichier_source' and c not in numeric_cols and c not in date_cols]

    return df, numeric_cols, date_cols, text_cols


def add_derived_features(df, numeric_cols, date_cols, text_cols):
    """
    Ajoute des colonnes dérivées pour que toutes les visualisations puissent fonctionner.
    """
    df = df.copy()
    new_text = []
    new_num = []

    if date_cols:
        d = date_cols[0]
        try:
            df[d] = pd.to_datetime(df[d], errors='coerce')
        except Exception:
            pass

        # Dérivées catégorielles
        if 'Mois' not in df.columns:
            df['Mois'] = df[d].dt.to_period('M').astype(str)
            new_text.append('Mois')
        if 'Année' not in df.columns:
            df['Année'] = df[d].dt.year.astype('Int64')
            df['Année'] = df['Année'].astype('string')
            new_text.append('Année')
        if 'JourSemaine' not in df.columns:
            jours_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            df['JourSemaine'] = df[d].dt.weekday.map(lambda i: jours_fr[int(i)] if pd.notna(i) else None)
            new_text.append('JourSemaine')
        if 'SemaineISO' not in df.columns:
            try:
                iso = df[d].dt.isocalendar()
                df['SemaineISO'] = (df[d].dt.year.astype('Int64').astype('string')
                                    + '-W' + iso.week.astype('Int64').astype('string'))
                new_text.append('SemaineISO')
            except Exception:
                pass

    # Numériques dérivées
    if len(numeric_cols) == 1:
        n = numeric_cols[0]
        if date_cols:
            df = df.sort_values(by=date_cols[0], kind='stable')
        cum_name = f'{n}_cumule'
        roll_name = f'{n}_rolling7'
        if cum_name not in df.columns:
            try:
                df[cum_name] = df[n].cumsum()
                new_num.append(cum_name)
            except Exception:
                pass
        if roll_name not in df.columns:
            try:
                df[roll_name] = df[n].rolling(window=7, min_periods=1).mean()
                new_num.append(roll_name)
            except Exception:
                pass

    # Mettre à jour les listes de colonnes
    text_cols = text_cols + [c for c in new_text if c not in text_cols]
    numeric_cols = numeric_cols + [c for c in new_num if c not in numeric_cols]

    return df, numeric_cols, date_cols, text_cols


def render_dashboard(df, key_prefix="gl"):
    """
    Affiche le tableau de bord complet (KPIs, Filtres, Graphes, PDF) pour un DataFrame donné.
    key_prefix permet de rendre unique les widgets (boutons, selects) streamlit.
    """
    # 1. Traitement des types
    df, numeric_cols, date_cols, text_cols = detect_column_types(df)
    df, numeric_cols, date_cols, text_cols = add_derived_features(df, numeric_cols, date_cols, text_cols)

    # 2. KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with kpi2:
        st.metric("📋 Colonnes", f"{len(df.columns)}")
    with kpi3:
        missing = (df.isna().sum().sum() / (len(df) * max(1, len(df.columns))) * 100)
        st.metric("⚠️ Manquant", f"{missing:.1f}%")
    with kpi4:
        st.metric("💾 Cellules", f"{len(df)*len(df.columns):,}")

    # 3. Filtres (Barre latérale ou Expander local pour ne pas polluer)
    # Pour les feuilles individuelles, on met les filtres dans un expander local au lieu de la sidebar
    # pour éviter de mélanger les filtres de globales et locales.
    
    df_filtered = df.copy()
    with st.expander(f"🔍 Filtres & Données ({key_prefix})", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            if date_cols:
                date_col = st.selectbox("📅 Date", ["Aucun"] + date_cols, key=f"{key_prefix}_date")
                if date_col != "Aucun":
                    min_d, max_d = df[date_col].min(), df[date_col].max()
                    if pd.notna(min_d) and pd.notna(max_d):
                        d_range = st.date_input("Période", [min_d, max_d], key=f"{key_prefix}_drange")
                        if len(d_range) == 2:
                            df_filtered = df_filtered[
                                (df_filtered[date_col] >= pd.Timestamp(d_range[0])) &
                                (df_filtered[date_col] <= pd.Timestamp(d_range[1]))
                            ]
        with f_col2:
            if text_cols:
                text_col = st.selectbox("🏷️ Catégorie", ["Aucun"] + text_cols, key=f"{key_prefix}_cat")
                if text_col != "Aucun":
                    opts = sorted(df_filtered[text_col].dropna().astype(str).unique())
                    sel = st.multiselect("Valeurs", opts, default=opts[:5], key=f"{key_prefix}_sel")
                    if sel:
                        df_filtered = df_filtered[df_filtered[text_col].astype(str).isin(sel)]
        with f_col3:
            if numeric_cols:
                num_col = st.selectbox("🔢 Valeur", ["Aucun"] + numeric_cols, key=f"{key_prefix}_num")
                if num_col != "Aucun":
                    mn, mx = float(df_filtered[num_col].min()), float(df_filtered[num_col].max())
                    if mn < mx:
                        rng = st.slider("Plage", mn, mx, (mn, mx), key=f"{key_prefix}_slid")
                        df_filtered = df_filtered[
                            (df_filtered[num_col] >= rng[0]) & 
                            (df_filtered[num_col] <= rng[1])
                        ]

        st.markdown("---")
        st.dataframe(df_filtered.head(100), use_container_width=True)
        
        # Download
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV", csv_data, f"data_{key_prefix}.csv", "text/csv", key=f"{key_prefix}_dl")

    # 4. Visualisations
    st.markdown("#### 📈 Visualisations")
    
    # Recalcul des colonnes après filtre (parfois utile)
    # numeric_cols etc sont déjà calculés sur le df complet, on les garde

    tabs = st.tabs(["Barres", "Courbes", "Camembert", "Nuage", "Heatmap"])
    
    # Config pour le bouton de téléchargement
    plotly_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': f'graphique_{key_prefix}',
            'height': 800,
            'width': 1200,
            'scale': 2 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    
    with tabs[0]: # Barres
        if text_cols and numeric_cols:
            c1, c2 = st.columns(2)
            bc = c1.selectbox("X", text_cols, key=f"{key_prefix}_bc")
            bv = c2.selectbox("Y", numeric_cols, key=f"{key_prefix}_bv")
            agg = df_filtered.groupby(bc)[bv].sum().sort_values(ascending=False).head(15).reset_index()
            st.plotly_chart(px.bar(agg, x=bc, y=bv, color=bv), use_container_width=True, key=f"{key_prefix}_chart_bar", config=plotly_config)
        else:
            st.info("Données insuffisantes")

    with tabs[1]: # Courbes
        if date_cols and numeric_cols:
            c1, c2 = st.columns(2)
            ld = c1.selectbox("Date", date_cols, key=f"{key_prefix}_ld")
            lv = c2.selectbox("Y", numeric_cols, key=f"{key_prefix}_lv")
            # Agg par jour par défaut
            agg = df_filtered.groupby(ld)[lv].sum().reset_index()
            st.plotly_chart(px.line(agg, x=ld, y=lv, markers=True), use_container_width=True, key=f"{key_prefix}_chart_line", config=plotly_config)
        else:
            st.info("Besoin d'une colonne Date et une Numérique")
            
    with tabs[2]: # Pie
        if text_cols:
            pc = st.selectbox("Catégorie", text_cols, key=f"{key_prefix}_pc")
            agg = df_filtered[pc].value_counts().head(10).reset_index()
            agg.columns = ['Cat', 'Val']
            st.plotly_chart(px.pie(agg, names='Cat', values='Val', hole=0.4), use_container_width=True, key=f"{key_prefix}_chart_pie", config=plotly_config)
        else:
             st.info("Pas de catégorie")

    with tabs[3]: # Scatter
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            sx = c1.selectbox("X", numeric_cols, key=f"{key_prefix}_sx")
            sy = c2.selectbox("Y", [c for c in numeric_cols if c!=sx], key=f"{key_prefix}_sy")
            dict_col = {}
            if text_cols:
                 dict_col['color'] = st.selectbox("Couleur", ["Aucune"]+text_cols, key=f"{key_prefix}_sc")
                 if dict_col['color'] == "Aucune": del dict_col['color']
            st.plotly_chart(px.scatter(df_filtered, x=sx, y=sy, **dict_col, opacity=0.7), use_container_width=True, key=f"{key_prefix}_chart_scatter", config=plotly_config)
        else:
            st.info("Besoin de 2 colonnes numériques")

    with tabs[4]: # Heatmap
        if len(text_cols) >= 2 and numeric_cols:
            hx = st.selectbox("X", text_cols, key=f"{key_prefix}_hx")
            hy = st.selectbox("Y", [c for c in text_cols if c!=hx], key=f"{key_prefix}_hy")
            hz = st.selectbox("Val", numeric_cols, key=f"{key_prefix}_hz")
            piv = df_filtered.pivot_table(index=hy, columns=hx, values=hz, aggfunc='sum').fillna(0)
            st.plotly_chart(px.imshow(piv, aspect='auto'), use_container_width=True, key=f"{key_prefix}_chart_heatmap", config=plotly_config)
        else:
            st.info("Besoin de 2 cat + 1 num")


# ========================================
# EXECUTION PRINCIPALE
# ========================================

if not all_sheets:
    st.info("👈 **Importez vos fichiers pour commencer**")
else:
    # 1. Affichage des feuilles individuelles (Analyses complètes)
    if all_sheets:
        st.subheader(f"📊 Analyse par feuille ({len(all_sheets)})")
        
        # On itère sur chaque feuille
        for name, sheet_df in all_sheets.items():
            # Nettoyage des caractères spéciaux pour la clé unique Streamlit
            safe_key = "".join([c for c in name if c.isalnum()])
            
            with st.expander(f"📑 Feuille : {name}", expanded=False):
                render_dashboard(sheet_df, key_prefix=f"sheet_{safe_key}")

    # 2. Analyse Globale (Fusion)
    st.markdown("---")
    st.header("🌍 Analyse Globale (Fusion)")
    
    # Gestion de l'état du bouton pour garder l'analyse affichée
    if 'fusion_active' not in st.session_state:
        st.session_state.fusion_active = False
        
    if st.button("🚀 Fusionner et Analyser tout"):
        st.session_state.fusion_active = True
        
    if st.session_state.fusion_active and all_sheets:
        # Création du DF fusionné
        dfs = list(all_sheets.values())
        global_df = pd.concat(dfs, ignore_index=True, sort=False)
        st.success(f"Fusion de {len(global_df):,} lignes effectuée !")
        render_dashboard(global_df, key_prefix="global")

