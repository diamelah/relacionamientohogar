import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import threading
import dash
from dash import dcc, html, dash_table, Input, Output
import json
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# Descargar stopwords si aún no se hizo
nltk.download('stopwords')

# Coordenadas de las provincias principales
provincias_coordenadas = {
    "BUENOS AIRES": [-34.6132, -58.3772],
    "ENTRE RIOS": [-31.7319, -60.5238],
    "CHACO": [-27.4514, -58.9868],
    "SANTA FE": [-31.6100, -60.7000],
    "CORDOBA": [-31.4167, -64.1833],
    "MENDOZA": [-32.8908, -68.8272],
    "CORRIENTES": [-27.4806, -58.8341],
    "SALTA": [-24.7821, -65.4232],
    "JUJUY": [-24.1858, -65.2995],
    "SAN JUAN": [-31.5375, -68.5364],
    "RIO NEGRO": [-41.1343, -69.3017],
    "TUCUMAN": [-26.8167, -65.2167],
    "NEUQUEN": [-38.9516, -68.0591],
    "LA RIOJA": [-29.4134, -66.8565],
    "FORMOSA": [-26.1849, -58.1731],
    "SAN LUIS": [-33.2950, -66.3356],
    "CATAMARCA": [-28.4696, -65.7795],
    "TIERRA DEL FUEGO": [-54.8019, -68.3029],
    "SANTIAGO DEL ESTERO": [-27.7834, -64.2642],
    "CHUBUT": [-43.3002, -65.1023],
    "LA PAMPA": [-36.6167, -64.2833],
    "MISIONES": [-27.3625, -55.8961]
}

# Sidebar: Carga del archivo CSV
st.sidebar.title("Carga del archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"], key="file_uploader_nps")

if uploaded_file is not None:
    # Cargar el archivo CSV y procesar la columna de fechas
    nps_data = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
    nps_data["Fecha"] = pd.to_datetime(nps_data["Fecha"]).dt.date

    # Sidebar: Filtros del Dashboard
    st.sidebar.title("Filtros del Dashboard")

    # Selección del periodo
    periodo_filtro = st.sidebar.selectbox("Seleccioná el tipo de rango de fecha:", ["Día", "Semana", "Mes"], index=0)

    if periodo_filtro == "Día":
        fecha_seleccionada = st.sidebar.date_input("Seleccioná un día específico:", value=nps_data["Fecha"].min())
        filtered_data = nps_data[nps_data["Fecha"] == fecha_seleccionada]

    elif periodo_filtro == "Semana":
        semana_inicio = st.sidebar.date_input("Seleccioná el inicio de la semana:", value=nps_data["Fecha"].min())
        semana_fin = semana_inicio + pd.Timedelta(days=6)
        st.sidebar.write(f"Mostrando datos desde {semana_inicio} hasta {semana_fin}.")
        filtered_data = nps_data[(nps_data["Fecha"] >= semana_inicio) & (nps_data["Fecha"] <= semana_fin)]

    else:  # Mes
        mes_seleccionado = st.sidebar.selectbox("Seleccioná un mes:", pd.date_range(nps_data["Fecha"].min(), nps_data["Fecha"].max(), freq="MS").strftime("%Y-%m"))
        mes_inicio = pd.to_datetime(mes_seleccionado).date()
        mes_fin = (pd.to_datetime(mes_seleccionado) + pd.DateOffset(months=1) - pd.DateOffset(days=1)).date()
        st.sidebar.write(f"Mostrando datos de {mes_inicio.strftime('%B %Y')}")
        filtered_data = nps_data[(nps_data["Fecha"] >= mes_inicio) & (nps_data["Fecha"] <= mes_fin)]

    st.title("Dashboard de Análisis de NPS - Relacionamiento Hogar")

    # 1. Previsualización de la tabla completa con filtros avanzados
    st.subheader("Tabla completa con filtros avanzados")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        opciones_grupo_nps = filtered_data["Grupo_NPS"].unique()
        grupo_nps_seleccionado = st.multiselect(
            "Filtrar por Grupo NPS:",
            options=opciones_grupo_nps,
            default=[opciones_grupo_nps[0]]
        )
    with col2:
        opciones_categoria = filtered_data["Categoria"].unique()
        categoria_seleccionada = st.multiselect(
            "Filtrar por Categoría:",
            options=opciones_categoria,
            default=[opciones_categoria[0]]
        )
    with col3:
        opciones_provincia = filtered_data["Provincia"].unique()
        provincia_seleccionada = st.multiselect(
            "Filtrar por Provincia:",
            options=opciones_provincia,
            default=[opciones_provincia[0]]
        )
    with col4:
        opciones_tecnologia = filtered_data["Tecnologia"].unique()
        tecnologia_seleccionada = st.multiselect(
            "Filtrar por Tecnología:",
            options=opciones_tecnologia,
            default=[opciones_tecnologia[0]]
        )
    tabla_filtrada = filtered_data[
        (filtered_data["Grupo_NPS"].isin(grupo_nps_seleccionado)) &
        (filtered_data["Categoria"].isin(categoria_seleccionada)) &
        (filtered_data["Provincia"].isin(provincia_seleccionada)) &
        (filtered_data["Tecnologia"].isin(tecnologia_seleccionada))
    ]
    st.dataframe(tabla_filtrada, height=400, use_container_width=True)

    # 2. Gráfico de torta con hover interactivo: Grupo_NPS y Categoría
    st.subheader("Distribución NPS por Grupos y Categorías")
    provincia_seleccionada_torta = st.selectbox(
        "Seleccioná una provincia para analizar:",
        options=["Todas"] + list(filtered_data["Provincia"].unique())
    )
    if provincia_seleccionada_torta != "Todas":
        datos_torta = filtered_data[filtered_data["Provincia"] == provincia_seleccionada_torta]
    else:
        datos_torta = filtered_data
    grupo_nps_counts = datos_torta.groupby("Grupo_NPS").size().reset_index(name="Cantidad")
    categoria_counts = datos_torta.groupby(["Grupo_NPS", "Categoria"]).size().reset_index(name="Cantidad")
    categoria_counts["Porcentaje"] = (categoria_counts["Cantidad"] / categoria_counts["Cantidad"].sum()) * 100
    colores_nps = {"Promotor": "#2ca02c", "Detractor": "#d62728", "Pasivo": "#1f77b4"}
    colores_categorias = {
        "Atencion al cliente": "#008000", 
        "Atencion Servicio Tecnico": "#228B22", 
        "Facturacion y Pago": "#FF0000"
    }
    fig = px.pie(
        grupo_nps_counts,
        values="Cantidad",
        names="Grupo_NPS",
        color="Grupo_NPS",
        color_discrete_map=colores_nps,
        hole=0.3
    )
    fig.update_traces(
        textinfo="label+percent",
        marker=dict(line=dict(color='black', width=1.5)),
        textfont_size=16
    )
    leyenda_html = "".join(
        [
            f"<div style='display: flex; align-items: center; margin-bottom: 3px;'>"
            f"<div style='width: 10px; height: 10px; background-color: {colores_nps.get(grupo, '#808080')}; margin-right: 5px; border: 1px solid black;'></div>"
            f"<span style='font-size: 12px;'>{cat} ({grupo}): {porc:.1f}%</span>"
            f"</div>"
            for grupo, cat, porc in zip(categoria_counts["Grupo_NPS"], categoria_counts["Categoria"], categoria_counts["Porcentaje"])
        ]
    )
    col1_chart, col2_chart = st.columns([2, 1])
    with col1_chart:
        st.plotly_chart(fig)
    with col2_chart:
        st.markdown(f"""<div style='border:1px solid black; padding:10px; background-color:white;'>
                    {leyenda_html}
                    </div>""", unsafe_allow_html=True)

    # 3. Gráfico de líneas para la evolución del NPS a lo largo del tiempo
    st.subheader("Tendencia de NPS a lo largo del tiempo")
    provincia_linea = st.selectbox(
        "Seleccione la Provincia para el análisis de tendencias:",
        options=filtered_data["Provincia"].unique(),
        index=0
    )
    data_filtrada_linea = filtered_data[filtered_data["Provincia"] == provincia_linea]
    grupo_nps_linea = st.multiselect(
        "Seleccione el Grupo NPS para el análisis:",
        options=["Promotor", "Detractor", "Pasivo"],
        default=["Promotor", "Detractor", "Pasivo"]
    )
    data_filtrada_linea = data_filtrada_linea[data_filtrada_linea["Grupo_NPS"].isin(grupo_nps_linea)]
    line_chart_data = data_filtrada_linea.groupby(["Fecha", "Grupo_NPS"]).size().reset_index(name="Count")
    line_chart = px.line(
        line_chart_data,
        x="Fecha",
        y="Count",
        color="Grupo_NPS",
        title=f"Evolución de NPS en {provincia_linea}",
        color_discrete_map={
            "Promotor": "green",
            "Detractor": "red",
            "Pasivo": "blue"
        }
    )
    st.plotly_chart(line_chart)

    # 4. Mapa de Calor de NPS por Provincias con opción de visualizar individual o combinación de grupos
    st.subheader("Mapa de Calor de NPS por Provincias")
    grupo_nps_opcion = st.selectbox(
        "Seleccione la visualización deseada:",
        options=["Todos", "Promotor", "Detractor", "Pasivo"],
        index=0
    )
    color_map = {"Promotor": "green", "Detractor": "red", "Pasivo": "blue"}
    if grupo_nps_opcion != "Todos":
        data_mapa = filtered_data[filtered_data["Grupo_NPS"] == grupo_nps_opcion]
        datos_por_provincia = data_mapa.groupby("Provincia").size().reset_index(name="Cantidad")
        datos_por_provincia = datos_por_provincia[datos_por_provincia["Provincia"].isin(provincias_coordenadas.keys())]
        datos_por_provincia["lat"] = datos_por_provincia["Provincia"].map(lambda x: provincias_coordenadas[x][0])
        datos_por_provincia["lon"] = datos_por_provincia["Provincia"].map(lambda x: provincias_coordenadas[x][1])
        mapa_provincias = px.scatter_mapbox(
            datos_por_provincia,
            lat="lat",
            lon="lon",
            size="Cantidad",
            hover_name="Provincia",
            hover_data={"Cantidad": True, "lat": False, "lon": False},
            size_max=30,
            zoom=4,
            mapbox_style="carto-positron",
            title=f"Distribución de {grupo_nps_opcion} por Provincia",
            height=600,
            width=1000
        )
        mapa_provincias.update_traces(marker=dict(color=color_map[grupo_nps_opcion]))
    else:
        data_mapa = filtered_data.copy()
        data_grouped = data_mapa.groupby(["Provincia", "Grupo_NPS"]).size().reset_index(name="Cantidad")
        data_grouped = data_grouped[data_grouped["Provincia"].isin(provincias_coordenadas.keys())]
        data_grouped["lat"] = data_grouped["Provincia"].map(lambda x: provincias_coordenadas[x][0])
        data_grouped["lon"] = data_grouped["Provincia"].map(lambda x: provincias_coordenadas[x][1])
        offsets = {
            "Promotor": (0.05, 0.05),
            "Detractor": (-0.05, -0.05),
            "Pasivo": (0.05, -0.05)
        }
        data_grouped["lat"] = data_grouped.apply(lambda row: row["lat"] + offsets[row["Grupo_NPS"]][0], axis=1)
        data_grouped["lon"] = data_grouped.apply(lambda row: row["lon"] + offsets[row["Grupo_NPS"]][1], axis=1)
        mapa_provincias = px.scatter_mapbox(
            data_grouped,
            lat="lat",
            lon="lon",
            size="Cantidad",
            color="Grupo_NPS",
            hover_name="Provincia",
            hover_data={"Cantidad": True, "lat": False, "lon": False},
            size_max=30,
            zoom=4,
            mapbox_style="carto-positron",
            title="Distribución de NPS por Provincia (Todos los Grupos)",
            height=600,
            width=1000,
            color_discrete_map=color_map
        )
    st.plotly_chart(mapa_provincias)

    # Detalle de NPS por Localidad
    provincia_seleccionada = st.selectbox("Seleccioná una provincia para ver el detalle de localidades:", options=filtered_data["Provincia"].unique())
    st.subheader(f"Detalle de NPS en localidades de {provincia_seleccionada}")
    detalle_localidades = filtered_data[filtered_data["Provincia"] == provincia_seleccionada].groupby(["Localidad", "Grupo_NPS", "Categoria"]).size().reset_index(name="Cantidad")
    total_por_localidad = detalle_localidades.groupby("Localidad")["Cantidad"].transform("sum")
    detalle_localidades["Porcentaje"] = (detalle_localidades["Cantidad"] / total_por_localidad) * 100
    detalle_localidades["Porcentaje"] = detalle_localidades["Porcentaje"].round(2)
    tabla_localidades = detalle_localidades.pivot(index=["Localidad", "Categoria"], columns="Grupo_NPS", values="Porcentaje").reset_index().fillna(0)
    st.dataframe(tabla_localidades)

    # Exportar datos a Excel
    def convertir_a_excel(df):
        output = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
        df.to_excel(output, sheet_name='Datos Filtrados', index=False)
        output.save()
        return output

    if st.button("Exportar datos filtrados a Excel"):
        convertir_a_excel(filtered_data)
        st.success("¡Datos exportados exitosamente a 'output.xlsx'!")
else:
    st.warning("Por favor, sube un archivo CSV para comenzar.")



