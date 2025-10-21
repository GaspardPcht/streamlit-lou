import streamlit as st
import pandas as pd
import numpy as np
import csv
import chardet
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="📊 Analyseur Marketing", layout="wide")

# CSS pour améliorer le rendu
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Analyseur de Données Marketing</p>', unsafe_allow_html=True)
st.write("**Importez vos fichiers CSV/XLSX — Visualisez instantanément vos données**")

# ========================================
# SECTION 1 : IMPORT MULTI-FICHIERS
# ========================================

st.sidebar.title("📁 Import de fichiers")
uploaded_files = st.sidebar.file_uploader(
    "Glissez vos fichiers ici (CSV ou XLSX)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Vous pouvez importer plusieurs fichiers en même temps"
)

@st.cache_data
def load_single_file(file_content, file_name):
    """Charge un fichier CSV ou Excel."""
    try:
        if file_name.lower().endswith('.csv'):
            # Détection encodage
            result = chardet.detect(file_content)
            encoding = result.get('encoding', 'utf-8')
            text = file_content.decode(encoding, errors='ignore')

            # Détection séparateur
            try:
                delimiter = csv.Sniffer().sniff(text[:5000]).delimiter
            except:
                delimiter = ',' if ',' in text else ';'

            return pd.read_csv(StringIO(text), sep=delimiter)
        else:
            return pd.read_excel(BytesIO(file_content), engine='openpyxl')
    except Exception as e:
        st.error(f"❌ Erreur {file_name}: {str(e)}")
        return None

# Chargement et fusion des fichiers
df = None
if uploaded_files:
    with st.spinner('⏳ Chargement des fichiers...'):
        dfs = []
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            loaded_df = load_single_file(file_content, uploaded_file.name)
            if loaded_df is not None:
                loaded_df['_fichier_source'] = uploaded_file.name
                dfs.append(loaded_df)

        if dfs:
            df = pd.concat(dfs, ignore_index=True, sort=False)
            st.sidebar.success(f"✅ {len(dfs)} fichier(s) chargé(s) — {len(df):,} lignes au total")
        else:
            st.sidebar.error("❌ Aucun fichier valide")

# ========================================
# SECTION 2 : TRAITEMENT DES DONNÉES
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

    # Conversion automatique des nombres (même s'ils sont stockés comme texte)
    for col in df.columns:
        if col == '_fichier_source':
            continue
        # Si la colonne est déjà numérique, on passe
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Si c'est une date, on passe aussi
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        # Essayer de convertir en numérique
        if df[col].dtype == 'object':
            try:
                # Nettoyer TOUS les caractères non numériques courants
                test_series = df[col].dropna().astype(str).head(100)
                cleaned = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace('\xa0', '', regex=False)  # Espace insécable
                    .str.replace(' ', '', regex=False)
                    .str.replace(',', '.', regex=False)   # Virgule décimale
                    .str.replace('€', '', regex=False)
                    .str.replace('$', '', regex=False)
                    .str.replace('£', '', regex=False)
                    .str.replace('%', '', regex=False)
                    .str.replace('−', '-', regex=False)   # Signe moins spécial
                    .str.replace('+', '', regex=False)
                )
                converted = pd.to_numeric(cleaned, errors='coerce')
                # Si au moins 50% des valeurs sont des nombres, on convertit
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

# ========================================
# SECTION 3 : FILTRES SIMPLES
# ========================================

def apply_filters(df, numeric_cols, date_cols, text_cols):
    """Applique des filtres simples et intuitifs."""
    df_filtered = df.copy()

    with st.sidebar.expander("🔍 Filtres", expanded=False):
        # Filtre par date
        if date_cols:
            date_col = st.selectbox("📅 Filtrer par date", ["Aucun"] + date_cols)
            if date_col != "Aucun":
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = st.date_input(
                        "Sélectionnez la période",
                        value=[min_date, max_date],
                        min_value=min_date,
                        max_value=max_date
                    )
                    if len(date_range) == 2:
                        df_filtered = df_filtered[
                            (df_filtered[date_col] >= pd.Timestamp(date_range[0])) &
                            (df_filtered[date_col] <= pd.Timestamp(date_range[1]))
                        ]

        # Filtre par catégorie
        if text_cols:
            text_col = st.selectbox("🏷️ Filtrer par catégorie", ["Aucun"] + text_cols)
            if text_col != "Aucun":
                unique_vals = df_filtered[text_col].dropna().astype(str).unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 100:
                    selected_vals = st.multiselect(
                        f"Valeurs de {text_col}",
                        options=sorted(unique_vals),
                        default=list(unique_vals[:10])
                    )
                    if selected_vals:
                        df_filtered = df_filtered[df_filtered[text_col].astype(str).isin(selected_vals)]

        # Filtre numérique
        if numeric_cols:
            num_col = st.selectbox("🔢 Filtrer par valeur", ["Aucun"] + numeric_cols)
            if num_col != "Aucun":
                min_val = float(df_filtered[num_col].min())
                max_val = float(df_filtered[num_col].max())
                if min_val < max_val:
                    val_range = st.slider(
                        f"Plage de {num_col}",
                        min_val, max_val, (min_val, max_val)
                    )
                    df_filtered = df_filtered[
                        (df_filtered[num_col] >= val_range[0]) &
                        (df_filtered[num_col] <= val_range[1])
                    ]

    return df_filtered

# ========================================
# AFFICHAGE PRINCIPAL
# ========================================

if df is None:
    st.info("👈 **Importez vos fichiers dans la barre latérale pour commencer**")
    st.markdown("""
    ### 🎯 Fonctionnalités :
    - ✅ Import multi-fichiers (CSV + XLSX)
    - ✅ Fusion automatique des données
    - ✅ Filtres intuitifs
    - ✅ Visualisations interactives (camemberts, barres, courbes, etc.)
    - ✅ Export PDF du rapport
    """)
else:
    # Traitement des données
    df, numeric_cols, date_cols, text_cols = detect_column_types(df)

    # KPIs en haut
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Lignes totales", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", f"{len(df.columns)-1}")
    with col3:
        missing = (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("⚠️ Données manquantes", f"{missing:.1f}%")
    with col4:
        if '_fichier_source' in df.columns:
            st.metric("📁 Fichiers", df['_fichier_source'].nunique())

    # Application des filtres
    df_filtered = apply_filters(df, numeric_cols, date_cols, text_cols)

    if len(df_filtered) < len(df):
        st.info(f"🔍 **{len(df_filtered):,}** lignes affichées sur **{len(df):,}** (filtre actif)")

    # Aperçu des données
    st.subheader("📋 Aperçu des données")
    st.dataframe(df_filtered.head(100), use_container_width=True)

    # Boutons de téléchargement
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger CSV",
            csv_data,
            "donnees.csv",
            "text/csv"
        )

    # ========================================
    # SECTION 4 : VISUALISATIONS
    # ========================================

    st.markdown("---")
    st.subheader("📈 Visualisations")

    viz_tabs = st.tabs([
        "🥧 Camembert",
        "📊 Barres",
        "📈 Courbes",
        "🌳 TreeMap",
        "☀️ Sunburst",
        "🔥 Heatmap",
        "💹 Nuage de points"
    ])

    # TAB 1: CAMEMBERT
    with viz_tabs[0]:
        st.markdown("### 🥧 Répartition par catégorie")
        if text_cols:
            col_pie1, col_pie2 = st.columns(2)
            with col_pie1:
                pie_col = st.selectbox("Choisir la catégorie", text_cols, key="pie_cat")
            with col_pie2:
                top_n = st.slider("Nombre de catégories", 5, 20, 10, key="pie_top")

            # Calcul des données
            pie_data = (
                df_filtered[pie_col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(top_n)
                .reset_index()
            )
            pie_data.columns = ['Catégorie', 'Valeur']

            if len(pie_data) > 0:
                # Graphique amélioré
                fig = px.pie(
                    pie_data,
                    names='Catégorie',
                    values='Valeur',
                    title=f"Top {top_n} - {pie_col}",
                    hole=0.3,  # Donut
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(
                    textposition='outside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='white', width=2))
                )
                fig.update_layout(
                    showlegend=True,
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tableau de données
                with st.expander("📊 Voir les chiffres détaillés"):
                    st.dataframe(pie_data, use_container_width=True)
            else:
                st.warning("Aucune donnée à afficher")
        else:
            st.info("Aucune colonne catégorielle détectée")

    # TAB 2: BARRES
    with viz_tabs[1]:
        st.markdown("### 📊 Graphique en barres")
        if text_cols and numeric_cols:
            col_bar1, col_bar2, col_bar3 = st.columns(3)
            with col_bar1:
                bar_cat = st.selectbox("Catégorie (X)", text_cols, key="bar_cat")
            with col_bar2:
                bar_val = st.selectbox("Valeur (Y)", numeric_cols, key="bar_val")
            with col_bar3:
                bar_top = st.slider("Top N", 5, 30, 15, key="bar_top")

            # Agrégation
            bar_data = (
                df_filtered.groupby(bar_cat)[bar_val]
                .sum()
                .sort_values(ascending=False)
                .head(bar_top)
                .reset_index()
            )

            if len(bar_data) > 0:
                fig = px.bar(
                    bar_data,
                    x=bar_cat,
                    y=bar_val,
                    title=f"{bar_val} par {bar_cat}",
                    color=bar_val,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée à afficher")
        else:
            st.info("Il faut au moins une colonne catégorielle et une colonne numérique")

    # TAB 3: COURBES
    with viz_tabs[2]:
        st.markdown("### 📈 Évolution temporelle")
        if date_cols and numeric_cols:
            col_line1, col_line2 = st.columns(2)
            with col_line1:
                line_date = st.selectbox("Date", date_cols, key="line_date")
            with col_line2:
                line_val = st.selectbox("Valeur", numeric_cols, key="line_val")

            # Agrégation par jour
            df_line = df_filtered.dropna(subset=[line_date]).copy()
            df_line['_date'] = pd.to_datetime(df_line[line_date]).dt.date
            line_data = df_line.groupby('_date')[line_val].sum().reset_index()

            if len(line_data) > 0:
                fig = px.line(
                    line_data,
                    x='_date',
                    y=line_val,
                    title=f"Évolution de {line_val}",
                    markers=True
                )
                fig.update_layout(hovermode='x unified', height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée à afficher")
        else:
            st.info("Il faut au moins une colonne de date et une colonne numérique")

    # TAB 4: TREEMAP
    with viz_tabs[3]:
        st.markdown("### 🌳 TreeMap (vision hiérarchique)")
        if text_cols and numeric_cols:
            col_tree1, col_tree2 = st.columns(2)
            with col_tree1:
                tree_cat = st.selectbox("Catégorie", text_cols, key="tree_cat")
            with col_tree2:
                tree_val = st.selectbox("Valeur", numeric_cols, key="tree_val")

            tree_data = (
                df_filtered.groupby(tree_cat)[tree_val]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )

            if len(tree_data) > 0:
                fig = px.treemap(
                    tree_data,
                    path=[tree_cat],
                    values=tree_val,
                    title=f"{tree_val} par {tree_cat}",
                    color=tree_val,
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée à afficher")
        else:
            st.info("Il faut au moins une colonne catégorielle et une colonne numérique")

    # TAB 5: SUNBURST
    with viz_tabs[4]:
        st.markdown("### ☀️ Sunburst (2 niveaux)")
        if len(text_cols) >= 2 and numeric_cols:
            col_sun1, col_sun2, col_sun3 = st.columns(3)
            with col_sun1:
                sun_cat1 = st.selectbox("Niveau 1", text_cols, key="sun_1")
            with col_sun2:
                sun_cat2 = st.selectbox("Niveau 2", [c for c in text_cols if c != sun_cat1], key="sun_2")
            with col_sun3:
                sun_val = st.selectbox("Valeur", numeric_cols, key="sun_val")

            sun_data = df_filtered.groupby([sun_cat1, sun_cat2])[sun_val].sum().reset_index()

            if len(sun_data) > 0:
                fig = px.sunburst(
                    sun_data,
                    path=[sun_cat1, sun_cat2],
                    values=sun_val,
                    title=f"{sun_val} : {sun_cat1} → {sun_cat2}",
                    color=sun_val,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée à afficher")
        else:
            st.info("Il faut au moins 2 colonnes catégorielles et 1 colonne numérique")

    # TAB 6: HEATMAP
    with viz_tabs[5]:
        st.markdown("### 🔥 Heatmap (corrélation)")
        if len(text_cols) >= 2 and numeric_cols:
            col_heat1, col_heat2, col_heat3 = st.columns(3)
            with col_heat1:
                heat_x = st.selectbox("Axe X", text_cols, key="heat_x")
            with col_heat2:
                heat_y = st.selectbox("Axe Y", [c for c in text_cols if c != heat_x], key="heat_y")
            with col_heat3:
                heat_z = st.selectbox("Valeur", numeric_cols, key="heat_z")

            pivot = df_filtered.pivot_table(
                index=heat_y,
                columns=heat_x,
                values=heat_z,
                aggfunc='sum',
                fill_value=0
            )

            # Limiter pour lisibilité
            if pivot.shape[0] > 15:
                pivot = pivot.iloc[:15]
            if pivot.shape[1] > 15:
                pivot = pivot.iloc[:, :15]

            if pivot.size > 0:
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='YlOrRd'
                ))
                fig.update_layout(
                    title=f"{heat_z} : {heat_y} vs {heat_x}",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Aucune donnée à afficher")
        else:
            st.info("Il faut au moins 2 colonnes catégorielles et 1 colonne numérique")

    # TAB 7: SCATTER
    with viz_tabs[6]:
        st.markdown("### 💹 Nuage de points")
        if len(numeric_cols) >= 2:
            col_scat1, col_scat2, col_scat3 = st.columns(3)
            with col_scat1:
                scat_x = st.selectbox("X", numeric_cols, key="scat_x")
            with col_scat2:
                scat_y = st.selectbox("Y", [c for c in numeric_cols if c != scat_x], key="scat_y")
            with col_scat3:
                scat_color = st.selectbox("Couleur", ["Aucune"] + text_cols, key="scat_color")

            # Échantillonnage pour performance
            sample_size = min(2000, len(df_filtered))
            df_sample = df_filtered.sample(sample_size) if len(df_filtered) > sample_size else df_filtered

            fig = px.scatter(
                df_sample,
                x=scat_x,
                y=scat_y,
                color=None if scat_color == "Aucune" else scat_color,
                title=f"{scat_y} vs {scat_x}",
                opacity=0.6
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Il faut au moins 2 colonnes numériques")

    # ========================================
    # SECTION 5 : EXPORT PDF
    # ========================================

    st.markdown("---")
    st.subheader("📄 Export du rapport")

    if st.button("🎯 Générer le rapport PDF", type="primary"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors

            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()

            # Titre
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=30
            )
            story.append(Paragraph("📊 Rapport d'Analyse Marketing", title_style))
            story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))

            # Résumé des données
            story.append(Paragraph("📋 Résumé des données", styles['Heading2']))
            summary_data = [
                ['Métrique', 'Valeur'],
                ['Nombre de lignes', f"{len(df_filtered):,}"],
                ['Nombre de colonnes', f"{len(df_filtered.columns)}"],
                ['Fichiers sources', f"{df_filtered['_fichier_source'].nunique() if '_fichier_source' in df_filtered.columns else 'N/A'}"],
                ['Période d\'analyse', f"{datetime.now().strftime('%B %Y')}"]
            ]

            summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))

            # Top catégories
            if text_cols:
                story.append(PageBreak())
                story.append(Paragraph(f"🏆 Top catégories - {text_cols[0]}", styles['Heading2']))
                top_cats = df_filtered[text_cols[0]].value_counts().head(10).reset_index()
                top_cats.columns = ['Catégorie', 'Nombre']

                cat_data = [['Catégorie', 'Nombre']]
                for _, row in top_cats.iterrows():
                    cat_data.append([str(row['Catégorie'])[:50], f"{row['Nombre']:,}"])

                cat_table = Table(cat_data, colWidths=[4*inch, 2*inch])
                cat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 11),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                story.append(cat_table)

            # Statistiques numériques
            if numeric_cols:
                story.append(PageBreak())
                story.append(Paragraph("📊 Statistiques des valeurs numériques", styles['Heading2']))

                stats_data = [['Colonne', 'Minimum', 'Maximum', 'Moyenne', 'Total']]
                for col in numeric_cols[:5]:  # Limiter à 5 colonnes
                    stats_data.append([
                        col[:30],
                        f"{df_filtered[col].min():.2f}",
                        f"{df_filtered[col].max():.2f}",
                        f"{df_filtered[col].mean():.2f}",
                        f"{df_filtered[col].sum():.2f}"
                    ])

                stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1.5*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(stats_table)

            # Générer le PDF
            doc.build(story)
            pdf_buffer.seek(0)

            st.success("✅ Rapport généré avec succès !")
            st.download_button(
                "📥 Télécharger le rapport PDF",
                pdf_buffer,
                f"rapport_marketing_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                "application/pdf"
            )

        except ImportError:
            st.error("❌ La bibliothèque reportlab n'est pas installée. Exécutez : pip install reportlab")
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération du PDF : {str(e)}")

st.markdown("---")
st.caption("💡 Astuce : Les filtres sont dans la barre latérale. Les graphiques s'adaptent automatiquement à vos données !")
