"""
generate-visualisation.py

Erstellt interaktive Visualisierungen (Plotly) auf Basis der zuvor
berechneten Seitenmetriken (letter_results.csv) und Metadaten.

Visualisierungen umfassen:

- Seitenformate
- historische Buchformate (Overlay)
- Textflächenanteil
- Papierfarbe (RGB-Raum)
- Briefe pro Jahr
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re


def letters_by_size(df: pd.DataFrame):
    """
    Scatterplot der Seitenmaße (Scan-Orientierung).

    Zusätzlich werden Referenzlinien eingezeichnet:
    - 4:5
    - 2:3
    - Goldener Schnitt
    """
    fig = px.scatter(
        df,
        x="page_w_mm",
        y="page_h_mm",
        title="Abmaße der Briefe",
        color="format_jl",
        hover_name="filename"
    )

    # Referenzlinien
    x_vals = np.linspace(0, 350, 100)

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=(5 / 4) * x_vals,
        mode="lines",
        line=dict(dash="dash"),
        name="4/5"
    ))

    fig.write_html(
        "./visualizations/Dehmel_letters-by-size.html"
    )


def extract_year(date):
    """
    Extrahiert das Jahr aus einem Datumsstring.

    Es werden nur Jahre > 1800 akzeptiert.
    """
    match = re.search(r"\d{4}", str(date))
    return (
        match.group(0)
        if match and int(match.group(0)) > 1800
        else None
    )


def letter_by_years(df):
    """
    Balkendiagramm der Briefanzahl pro Jahr.
    """
    yearly_counts = df["Jahr"].value_counts().reset_index()
    yearly_counts.columns = ["Jahr", "Anzahl"]
    yearly_counts = yearly_counts.sort_values("Jahr")

    fig = px.bar(
        yearly_counts,
        x="Jahr",
        y="Anzahl",
        title="Anzahl der Briefe pro Jahr"
    )

    fig.write_html(
        "./visualizations/Dehmel_letters-by-year.html"
    )


def letters_by_color(df):
    """
    3D-Scatterplot der Papierfarbe im RGB-Raum.
    """
    df["color_rgb"] = df.apply(
        lambda row:
        f"rgb({row['bgr_color_r']},"
        f"{row['bgr_color_g']},"
        f"{row['bgr_color_b']})",
        axis=1
    )

    fig = px.scatter_3d(
        df,
        x="bgr_color_r",
        y="bgr_color_g",
        z="bgr_color_b",
        title="Papierfarbe der Briefe"
    )

    fig.update_traces(
        marker=dict(color=df['color_rgb'], size=3)
    )

    fig.write_html(
        "./visualizations/Dehmel_letters-by-color.html"
    )


def main():
    """
    Hauptfunktion:

    1. Laden der Analyseergebnisse
    2. Zusammenführen mit Metadaten
    3. Datenbereinigung
    4. Erzeugung aller Visualisierungen
    """
    df_1 = pd.read_csv(
        r"./eval/letter_results.csv",
        sep=";"
    )

    df_2 = pd.read_csv(
        r"./metadata_total.csv",
        sep=",",
        encoding="UTF-8"
    )

    df = df_1.merge(
        df_2,
        left_on="hans_id_norm",
        right_on="hans_id"
    )

    df["Jahr"] = df[
        "Datum (zuerst lt Datumszeile, alternativ Poststempel)"
    ].apply(extract_year)

    # Ausschluss unsicherer DPI-Werte
    df = df[df["img_dpi"] != 72]

    letters_by_size(df)
    letters_by_color(df)
    letter_by_years(df)


main()
