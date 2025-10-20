
import streamlit as st
import pandas as pd
import numpy as np
import csv
import chardet
from io import StringIO, BytesIO
import plotly.express as px
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(page_title="Explorateur de données", layout="wide")
st.title("Explorateur de Données")
st.write("Importer un CSV ou XLSX, filtrer les données, créer des graphiques et générer un rapport PDF.")

st.sidebar.header("Importer le fichier")
uploaded_file = st.sidebar.file_uploader("Choisir un fichier CSV ou XLSX", type=['csv','xls','xlsx'])
df = None
if uploaded_file:
    try:
        data_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith('.csv'):
            result = chardet.detect(data_bytes)
            encoding = result['encoding'] or 'utf-8'
            text = data_bytes.decode(encoding, errors='ignore')
            try:
                sample = text[:5000]
                delimiter = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                delimiter = ';' if ';' in text else ','
            df = pd.read_csv(StringIO(text), sep=delimiter)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.sidebar.success("Fichier chargé avec succès !")
    except Exception as e:
        st.sidebar.error(f"Erreur lecture fichier : {e}")

if df is not None:
    st.subheader("Aperçu des données")
    st.write(df.head(10))
    st.write("Types de colonnes :")
    st.write(df.dtypes)

    # Filtres dynamiques
    df_filtered = df.copy()
    st.sidebar.header("Filtres")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            mn, mx = df[col].min(), df[col].max()
            if pd.notnull(mn) and pd.notnull(mx) and mn != mx:
                vals = st.sidebar.slider(col, float(mn), float(mx), (float(mn), float(mx)))
                df_filtered = df_filtered[df_filtered[col].between(vals[0], vals[1])]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dmin, dmax = df[col].min(), df[col].max()
            if pd.notnull(dmin) and pd.notnull(dmax):
                vals = st.sidebar.date_input(col, [dmin, dmax])
                if len(vals) == 2:
                    df_filtered = df_filtered[(df_filtered[col] >= np.datetime64(vals[0])) & (df_filtered[col] <= np.datetime64(vals[1]))]
        else:
            unique_vals = df[col].dropna().unique().tolist()
            if unique_vals:
                selected = st.sidebar.multiselect(col, unique_vals, default=unique_vals)
                if selected:
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]

    # Sélection du type de graphique
    st.subheader("Construction de graphique")
    chart_type = st.selectbox("Type de graphique", ["Pie Chart", "Bar Chart", "Line / Area Chart", "Scatter Plot", "Histogram", "Pivot Table"])
    fig = None

    if chart_type == "Pie Chart":
        cat_col = st.selectbox("Catégorie", [c for c in df_filtered.columns if df_filtered[c].dtype == object])
        val_col = st.selectbox("Valeur (numérique, optionnel)", [None] + [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        if cat_col:
            if val_col:
                data = df_filtered.groupby(cat_col)[val_col].sum().reset_index()
                fig = px.pie(data, names=cat_col, values=val_col, title=f"Camembert de {val_col} par {cat_col}")
            else:
                data = df_filtered[cat_col].value_counts().reset_index()
                data.columns = [cat_col, 'count']
                fig = px.pie(data, names=cat_col, values='count', title=f"Camembert de {cat_col}")

    elif chart_type == "Bar Chart":
        x_col = st.selectbox("X (catégoriel)", [c for c in df_filtered.columns if df_filtered[c].dtype == object])
        y_col = st.selectbox("Y (numérique)", [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        agg = st.selectbox("Agrégation", ["count", "sum", "mean", "median", "min", "max"])
        group_cols = st.multiselect("Regrouper (optionnel)", [c for c in df_filtered.columns if df_filtered[c].dtype == object], default=[])
        if x_col and y_col:
            cols = [x_col] + group_cols if group_cols else [x_col]
            grouped = df_filtered.groupby(cols)[y_col]
            if agg == "count":
                agg_df = grouped.count().reset_index(name='count')
                fig = px.bar(agg_df, x=x_col, y='count', color=group_cols[0] if group_cols else None, title=f"Barres (count) de {x_col}")
            else:
                if agg == "sum":
                    agg_df = grouped.sum().reset_index(name='sum'); ytitle='sum'
                elif agg == "mean":
                    agg_df = grouped.mean().reset_index(name='mean'); ytitle='mean'
                elif agg == "median":
                    agg_df = grouped.median().reset_index(name='median'); ytitle='median'
                elif agg == "min":
                    agg_df = grouped.min().reset_index(name='min'); ytitle='min'
                elif agg == "max":
                    agg_df = grouped.max().reset_index(name='max'); ytitle='max'
                fig = px.bar(agg_df, x=x_col, y=ytitle, color=group_cols[0] if group_cols else None, title=f"Barres ({agg}) de {y_col} par {x_col}")

    elif chart_type == "Line / Area Chart":
        x_options = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c]) or np.issubdtype(df_filtered[c].dtype, np.datetime64)]
        x_col = st.selectbox("X (num/datetime)", x_options)
        y_col = st.selectbox("Y (numérique)", [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        mode = st.radio("Mode", ["lines", "lines+markers", "area"])
        if x_col and y_col:
            df_line = df_filtered.sort_values(x_col)
            if mode == "area":
                fig = px.area(df_line, x=x_col, y=y_col, title=f"Aire de {y_col} vs {x_col}")
            else:
                fig = px.line(df_line, x=x_col, y=y_col, title=f"Lignes de {y_col} vs {x_col}", markers=(mode=="lines+markers"))

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("X (numérique)", [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        y_col = st.selectbox("Y (numérique)", [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        size_col = st.selectbox("Taille (optionnel)", [None] + [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        color_col = st.selectbox("Couleur (optionnel)", [None] + list(df_filtered.columns))
        if x_col and y_col:
            fig = px.scatter(df_filtered, x=x_col, y=y_col,
                             size=size_col if size_col else None,
                             color=color_col if color_col else None,
                             title=f"Nuage de points de {y_col} vs {x_col}")

    elif chart_type == "Histogram":
        h_col = st.selectbox("Colonne numérique", [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        bins = st.slider("Nombre de bins", 1, 100, 10)
        if h_col:
            fig = px.histogram(df_filtered, x=h_col, nbins=bins, title=f"Histogramme de {h_col}")

    elif chart_type == "Pivot Table":
        group_cols = st.multiselect("Regrouper par", df_filtered.columns.tolist())
        pivot_col = st.selectbox("Colonne calculée (numérique)", [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])])
        agg_func = st.selectbox("Fonction", ["sum", "count", "mean", "median", "min", "max", "pct"])
        if group_cols and pivot_col:
            grouped = df_filtered.groupby(group_cols)[pivot_col]
            if agg_func == "sum":
                pivot_df = grouped.sum().reset_index(name='sum')
            elif agg_func == "count":
                pivot_df = grouped.count().reset_index(name='count')
            elif agg_func == "mean":
                pivot_df = grouped.mean().reset_index(name='mean')
            elif agg_func == "median":
                pivot_df = grouped.median().reset_index(name='median')
            elif agg_func == "min":
                pivot_df = grouped.min().reset_index(name='min')
            elif agg_func == "max":
                pivot_df = grouped.max().reset_index(name='max')
            elif agg_func == "pct":
                pivot_df = grouped.sum().reset_index(name='sum')
                total = pivot_df['sum'].sum()
                pivot_df['% du total'] = pivot_df['sum'] / total * 100
            st.write("Tableau croisé :")
            st.write(pivot_df)

    # Affichage du graphique
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        # Télécharger PNG
        img_buf = BytesIO()
        fig.write_image(img_buf, format='png')
        st.download_button("Télécharger le graphique (PNG)", data=img_buf.getvalue(), file_name="graphique.png", mime="image/png")

        # Ajouter au rapport PDF
        if 'charts' not in st.session_state:
            st.session_state.charts = []
        if 'captions' not in st.session_state:
            st.session_state.captions = []
        caption = fig.layout.title.text if fig.layout.title.text else chart_type
        if st.button("Ajouter au rapport PDF"):
            st.session_state.charts.append(fig)
            st.session_state.captions.append(caption)
            st.success("Graphique ajouté au rapport")

    # Générer rapport PDF
    if st.session_state.get('charts'):
        if st.button("Générer et télécharger le rapport PDF"):
            pdf_buf = BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            # Titre et date
            story.append(Paragraph("Rapport de visualisation de données", styles['Title']))
            story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles['Normal']))
            # Résumé des données
            story.append(Paragraph(f"Nombre de lignes : {df.shape[0]}, nombre de colonnes : {df.shape[1]}.", styles['Normal']))
            story.append(PageBreak())
            # Ajout des graphiques
            for fig, cap in zip(st.session_state.charts, st.session_state.captions):
                story.append(Paragraph(cap, styles['Heading2']))
                img_data = BytesIO()
                fig.write_image(img_data, format='png')
                img_data.seek(0)
                im = Image(img_data, width=6*inch, height=4*inch)
                story.append(im)
                story.append(PageBreak())
            doc.build(story)
            st.download_button("Télécharger le rapport PDF", data=pdf_buf.getvalue(), file_name="rapport.pdf", mime="application/pdf")
