import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re


def letters_by_size(df: pd.DataFrame):
    """plots documents by raw size in scan orientation"""
    fig = px.scatter(df, x="page_w_mm", y="page_h_mm", title="Abmaße der Briefe (in Schreibrichtung)",
                     color="format_jl", hover_name="filename",
                     hover_data=["Absender:innen", "Datum (zuerst lt Datumszeile, alternativ Poststempel)", "Transkribus-ID"],
                     labels={"page_w_mm": "Breite (mm)", "page_h_mm": "Höhe (mm)", "format_jl": "Format"})

    # dotted lines for ratio comparison
    x_vals_4_5 = np.linspace(0,350,100)
    y_vals_4_5 = (5/4) * x_vals_4_5
    x_vals_2_3 = np.linspace(0,350,100)
    y_vals_2_3 = (3/2) *x_vals_2_3
    x_vals_golden = np.linspace(0,350,100)
    y_vals_golden = ((1 + np.sqrt(5)) / 2) * x_vals_golden

    fig.add_trace(go.Scatter(x=x_vals_4_5, y=y_vals_4_5, mode="lines", line=dict(dash="dash", color="blue", width=2), name="4/5"))
    fig.add_trace(go.Scatter(x=x_vals_2_3, y=y_vals_2_3, mode="lines", line=dict(dash="dash", color="red", width=2), name="2/3"))
    fig.add_trace(go.Scatter(x=x_vals_golden, y=y_vals_golden, mode="lines", line=dict(dash="dash", color="orange", width=2), name="Golden Ratio"))

    fig.write_html("./visualizations/Dehmel_letters-by-size.html")


def letters_by_format(df: pd.DataFrame):
    """plots documents by format of shorter and longer side, scan orientation is ignored"""
    fig = px.scatter(df, x="page_shorter_mm", y="page_longer_mm", title="Papierformat der Briefe",
                     color="format_jl", hover_name="filename",
                     hover_data=["Absender:innen", "Datum (zuerst lt Datumszeile, alternativ Poststempel)", "Transkribus-ID"],
                     labels={"page_shorter_mm": "Kürzere Seite (mm)", "page_longer_mm": "Längere Seite (mm)", "format_jl": "Format"})

    x_vals_4_5 = np.linspace(0,250,100)
    y_vals_4_5 = (5/4) * x_vals_4_5
    x_vals_2_3 = np.linspace(0,350,100)
    y_vals_2_3 = (3/2) *x_vals_2_3
    x_vals_golden = np.linspace(0,350,100)
    y_vals_golden = ((1 + np.sqrt(5)) / 2) * x_vals_golden

    # overlay: historical book formats, extracted from Krebs
    fig.add_hrect(y0=350, y1=250, fillcolor="gray", opacity=0.2, line_width=0, annotation_text="Quart 4°", annotation_position="top right")
    fig.add_hrect(y0=250, y1=225, fillcolor="purple", opacity=0.2, line_width=0, annotation_text="Groß-Oktav Gr.-8°", annotation_position="top right")
    fig.add_hrect(y0=225, y1=185, fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Oktav 8°", annotation_position="top right")
    fig.add_hrect(y0=185, y1=170, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Klein-Oktav Kl.-8°", annotation_position="top right")
    fig.add_hrect(y0=170, y1=150, fillcolor="blue", opacity=0.2, line_width=0, annotation_text="Duodez 12°", annotation_position="top right")
    fig.add_hrect(y0=150, y1=130, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Sedez 16°", annotation_position="top right")

    fig.add_trace(go.Scatter(x=x_vals_4_5, y=y_vals_4_5, mode="lines", line=dict(dash="dash", color="blue", width=2), name="4/5"))
    fig.add_trace(go.Scatter(x=x_vals_2_3, y=y_vals_2_3, mode="lines", line=dict(dash="dash", color="red", width=2), name="2/3"))
    fig.add_trace(go.Scatter(x=x_vals_golden, y=y_vals_golden, mode="lines", line=dict(dash="dash", color="orange", width=2), name="Goldener Schnitt"))


    # Zeige das Diagramm an
    fig.write_html("./visualizations/Dehmel_letters-by-format.html")


def letters_by_format_wo_overlay(df):
    """plots documents by format of shorter and longer side, scan orientation is ignored, without overlays"""
    fig = px.scatter(df, x="page_shorter_mm", y="page_longer_mm", title="Papierformate der Briefe",
                     color="format_jl", hover_name="filename",
                     hover_data=["Absender:innen", "Datum (zuerst lt Datumszeile, alternativ Poststempel)", "Transkribus-ID"],
                     labels={"page_w_mm": "Breite (mm)", "page_h_mm": "Höhe (mm)", "format_jl": "Format"})

    x_vals_4_5 = np.linspace(0,350,100)
    y_vals_4_5 = (5/4) * x_vals_4_5
    x_vals_2_3 = np.linspace(0,350,100)
    y_vals_2_3 = (3/2) *x_vals_2_3
    x_vals_golden = np.linspace(0,350,100)
    y_vals_golden = ((1 + np.sqrt(5)) / 2) * x_vals_golden

    fig.add_trace(go.Scatter(x=x_vals_4_5, y=y_vals_4_5, mode="lines", line=dict(dash="dash", color="blue", width=2), name="4/5"))
    fig.add_trace(go.Scatter(x=x_vals_2_3, y=y_vals_2_3, mode="lines", line=dict(dash="dash", color="red", width=2), name="2/3"))
    fig.add_trace(go.Scatter(x=x_vals_golden, y=y_vals_golden, mode="lines", line=dict(dash="dash", color="orange", width=2), name="Golden Ratio"))

    # Zeige das Diagramm an
    fig.write_html("./visualizations/Dehmel_letters-by-format_without-overlay.html")


def letters_by_format_only_letters_postcards_telegrams(df):
    """plots documents by format of shorter and longer side, scan orientation is ignored, only for certain document
    types (postcards, letters, telegrams) and with box plots in marginals"""
    df_filtered = df[df['format_jl'].isin(['Brief', 'Postkarte', 'Ansichtskarte', 'Telegramm'])]
    df_filtered["format_jl"] = df_filtered["format_jl"].replace("Ansichtskarte", "Postkarte")
    fig = px.scatter(df_filtered, x="page_shorter_mm", y="page_longer_mm", title="Papierformate der Dokumente (Ansichtskarten werden als Postkarten gezählt)",
                     color="format_jl", hover_name="hans_id_norm", marginal_x="box", marginal_y="box",
                     hover_data=["Absender:innen", "Datum (zuerst lt Datumszeile, alternativ Poststempel)",
                                 "Transkribus-ID"],
                     labels={"page_shorter_mm": "Kürzere Seite (mm)", "page_longer_mm": "längere Seite (mm)", "format_jl": "Format"})
    fig.write_html("./visualizations/Dehmel_letters-by-format_only-letters-postcards-telegrams.html")


def letters_by_sender (df):
    """plots documents by sender in scan orientation"""
    df["main_sender"] = df["Absender:innen"].str.strip(" [].,1234567890()").str.split("[,;]").str[0]
    fig = px.scatter(df, x="page_w_mm", y="page_h_mm", title="Absender:innen der Briefe",
                     color="main_sender", hover_name="hans_id_norm",
                     hover_data=["Absender:innen", "format_jl", "Datum (zuerst lt Datumszeile, alternativ Poststempel)",
                                 "Transkribus-ID"],
                     labels={"page_w_mm": "Breite (mm)", "page_h_mm": "Höhe (mm)", "format_jl": "Format"})
    # Zeige das Diagramm an
    fig.write_html("./visualizations/Dehmel_letters-by-sender.html")


def page_fill (df):
    fig = px.box(df[(df["format_jl"] == "Brief")], y="text_fill", title="Schriftraum der Briefseiten")
    mean_fischer = df[(df['Absender:innen'] == "Samuel Fischer") & (df["format_jl"] == "Brief")]['text_fill'].mean()
    mean_r_dehmel = df[(df['Absender:innen'] == "Richard Dehmel") & (df["format_jl"] == "Brief")]['text_fill'].mean()
    mean_i_dehmel = df[(df['Absender:innen'] == "Ida Dehmel") & (df["format_jl"] == "Brief")]['text_fill'].mean()
    fig.add_hline(y=mean_fischer, line=dict(color='red', dash='dash', width=2), name="Samuel Fischer")
    fig.add_annotation(x=-.3, y=mean_fischer, text="Samuel Fischer", bgcolor="lightpink", opacity=0.7, align="center")
    fig.add_hline(y=mean_r_dehmel, line=dict(color='green', dash='dash', width=2), name="Richard Dehmel")
    fig.add_annotation(x=-.3, y=mean_r_dehmel, text="Richard Dehmel", bgcolor="lightgreen", opacity=0.7, align="center")
    fig.add_hline(y=mean_i_dehmel, line=dict(color='yellow', dash='dash', width=2), name="Ida Dehmel")
    fig.add_annotation(x=-.3, y=mean_i_dehmel, text="Ida Dehmel", bgcolor="lightyellow", opacity=0.7, align="center")
    fig.write_html("./visualizations/Dehmel_letters-by-text-fill.html")


def letters_by_color(df):
    """plots documents by color in 3D RGB space"""
    df["color_rgb"] = df.apply(lambda row: f"rgb({row['bgr_color_r']},{row['bgr_color_g']},{row['bgr_color_b']})", axis=1)
    fig = px.scatter_3d(df,
                        x="bgr_color_r", y="bgr_color_g", z="bgr_color_b",
                        labels={"bgr_color_r": "Rot", "bgr_color_g": "Grün", "bgr_color_b": "Blau"},
                        hover_name="filename",
                        hover_data=["Absender:innen", "format_jl",
                                    "Datum (zuerst lt Datumszeile, alternativ Poststempel)",
                                    "Transkribus-ID"],
                        title="Papierfarbe der Briefe")
    fig.update_traces(marker=dict(color=df['color_rgb'], size=3))
    fig.add_trace(go.Scatter3d(x=[0,255], y=[0,255], z=[0,255], mode="lines", line=dict(dash="dash", color="black", width=2), name="b/w"))
    fig.write_html("./visualizations/Dehmel_letters-by-color.html")


def letter_by_years(df):
    """plots documents by year as a bar chart"""
    yearly_counts = df["Jahr"].value_counts().reset_index()
    yearly_counts.columns = ["Jahr", "Anzahl"]
    yearly_counts = yearly_counts.sort_values("Jahr")
    fig = px.bar(yearly_counts, x="Jahr", y="Anzahl", title="Anzahl der Briefe pro Jahr")
    fig.write_html("./visualizations/Dehmel_letters-by-year.html")


def extract_year(date):
    """extracts year from date string"""
    # print(date)
    match = re.search(r"\d{4}", str(date))
    # print(match.group(0) if match else None)
    return match.group(0) if match and int(match.group(0)) > 1800 else None


def main():
    # data preprocessing
    df_1 = pd.read_csv(r"./eval/letter_results.csv", sep=";")
    df_1["hans_id_norm"] = df_1["hans_id"].str.replace("_duplicated", "")
    df_1.rename(columns = {"Unnamed: 0": "filename"}, inplace=True)
    df_2 = pd.read_csv(r"./metadata_total.csv", sep=",", encoding="UTF-8")
    df_2.rename(columns = {"Unnamed: 0": "hans_id", "Art des Dokuments (Brief/Postkarte/Feldpostkarte/Telegramm/Skizze/Beilage/...)": "format_jl"}, inplace=True)
    df_2 = df_2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df = df_1.merge(df_2, left_on="hans_id_norm", right_on="hans_id")
    df["Jahr"] = df["Datum (zuerst lt Datumszeile, alternativ Poststempel)"].apply(extract_year)
    # Ausnahme der Bilder mit einer DPI von 72, für diese sind keine gesicherten Maße bekannt
    df = df[df["img_dpi"] != 72]
    letters_by_size(df)
    letters_by_format(df)
    letters_by_sender(df)
    letters_by_color(df)
    letters_by_format_wo_overlay(df)
    letter_by_years(df)
    letters_by_format_only_letters_postcards_telegrams(df)
    page_fill(df)


main()