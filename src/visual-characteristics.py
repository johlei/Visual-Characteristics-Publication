"""
visual-characteristics.py

Dieses Modul analysiert gescannte historische Briefe (TIFF-Dateien) in
Kombination mit PAGE-XML-Dateien (Transkribus-Export) und extrahiert
visuelle Merkmale wie:

- Seitenmaße (in mm)
- Textflächenanteil
- Zeilenmetriken (Höhe, Breite, Fläche)
- Seitenränder
- Header-Fläche
- Papierfarbe (RGB)
- statistische Ausreißer

Die Ergebnisse werden als CSV-Dateien exportiert.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import statistics
import math

import cv2
import numpy as np
import pandas as pd
from lxml import etree
from alive_progress import alive_bar
from PIL import Image
from shapely.geometry import Polygon


def read_file_structure(dir_path: Path) -> dict[str, dict[str, list[Path]]]:
    """
    Durchsucht die Verzeichnisstruktur und sammelt TIFF- und XML-Dateien.

    Erwartete Struktur:
        Oberordner/
            Transkribus_Export/
                Hans_ID/
                    *.tif
                    page/*.xml

    Parameters
    ----------
    dir_path : Path
        Basisverzeichnis mit den Briefdaten.

    Returns
    -------
    dict
        Dictionary mit:
            {
                hans_id: {
                    "tif": [Liste von TIFF-Pfaden],
                    "xml": [Liste von XML-Pfaden]
                }
            }
    """
    result = {}

    for transkribus_folder in dir_path.iterdir():
        for hans_folder in transkribus_folder.iterdir():
            if hans_folder.is_dir():
                images = sorted(
                    (file for file in hans_folder.rglob("*.tif")
                     if file.is_file()),
                    key=lambda f: f.stem
                )
                xml_files = sorted(
                    (file for file in
                     hans_folder.rglob("./page/*.xml")
                     if file.is_file()),
                    key=lambda f: f.stem
                )
                result[hans_folder.name] = {
                    'tif': images,
                    'xml': xml_files
                }

    return result


def parse_xml(file_path: Path) -> etree.Element:
    """
    Parst eine PAGE-XML-Datei.

    Parameters
    ----------
    file_path : Path
        Pfad zur XML-Datei.

    Returns
    -------
    etree.Element
        Root-Element des XML-Dokuments.
    """
    try:
        pxml = etree.parse(file_path).getroot()
    except TypeError:
        print(f"XML syntax error in file {file_path}")
        pxml = etree.Element("None")

    return pxml


def extract_image_dpi(tif_path: Path) -> int:
    """
    Liest die DPI-Information aus einer TIFF-Datei.

    Falls horizontale und vertikale DPI unterschiedlich sind,
    werden die horizontalen DPI als Heuristik verwendet.

    Parameters
    ----------
    tif_path : Path

    Returns
    -------
    int
        DPI-Wert
    """
    with Image.open(tif_path) as img:
        dpi = img.info.get("dpi", (0, 0))

        if dpi[0] == dpi[1]:
            return dpi[0]

        print("ACHTUNG: Unterschiedliche DPI-Werte erkannt.")
        return dpi[0]


def pix_to_mm(length_pixel: int, dpi: int) -> float:
    """
    Konvertiert Pixel in Millimeter.

    Parameters
    ----------
    length_pixel : int
    dpi : int

    Returns
    -------
    float
    """
    return (length_pixel * 25.4) / dpi


def calculate_color(img, x_min, x_max, y_min, y_max) -> Tuple:
    """
    Berechnet den durchschnittlichen Farbwert (RGB) im Seitenbereich.

    Returns
    -------
    Tuple (R, G, B)
    """
    roi = img[y_min:y_max, x_min:x_max]
    mean_color_bgr = cv2.mean(roi)

    # OpenCV verwendet BGR → Umwandlung in RGB
    return mean_color_bgr[2], mean_color_bgr[1], mean_color_bgr[0]


def sort_points_by_angle(points):
    """
    Sortiert Polygonpunkte nach Winkel relativ zum Schwerpunkt.

    Wird benötigt, um eine konsistente Polygonfläche berechnen zu können.
    """
    centroid_x = sum(x for x, _ in points) / len(points)
    centroid_y = sum(y for _, y in points) / len(points)

    def angle_from_centroid(point):
        x, y = point
        return math.atan2(y - centroid_y, x - centroid_x)

    return sorted(points, key=angle_from_centroid)


def calculate_polygon_area(coord_list, dpi):
    """
    Berechnet die Fläche eines Polygons in mm².

    Parameters
    ----------
    coord_list : list
        Liste von (x,y)-Koordinaten.
    dpi : int

    Returns
    -------
    float
        Fläche in mm²
    """
    sorted_coords = sort_points_by_angle(coord_list)
    area_px = Polygon(sorted_coords).area

    # Umrechnung px² → mm²
    area_mm2 = ((25.4 / dpi) ** 2) * area_px

    return area_mm2


def search_for_outliers(data: Dict, key_name: str,
                        num_stdev: float) -> Dict:
    """
    Identifiziert Ausreißer anhand eines Z-Scores.

    Parameters
    ----------
    data : Dict
    key_name : str
        Attribut, das geprüft werden soll.
    num_stdev : float
        Schwellenwert in Standardabweichungen.

    Returns
    -------
    Dict
        {key: True/False}
    """
    values = [item[key_name] for item in data.values()]

    if len(values) < 2:
        return {}

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0:
        return {}

    return {
        key: abs((item[key_name] - mean) / stdev) > num_stdev
        for key, item in data.items()
    }


def calculate_fill(text_lines, text_regions,
                   page_w_mm, page_h_mm,
                   addition_mm2) -> float:
    """
    Berechnet den prozentualen Textflächenanteil einer Seite.

    Falls Zeilenfläche > Seitenfläche (Fehlerfall),
    wird stattdessen auf TextRegionen zurückgegriffen.
    """
    page_area_mm2 = page_h_mm * page_w_mm
    text_area_mm2 = sum(
        float(text_lines[i]["area_mm2"])
        for i in text_lines.keys()
    )

    if page_area_mm2 <= text_area_mm2:
        region_area = sum(
            float(text_regions[i]["area_mm2"])
            for i in text_regions.keys()
        )
        area_percentage = (region_area + addition_mm2) / page_area_mm2
    else:
        area_percentage = (text_area_mm2 + addition_mm2) / page_area_mm2

    return min(area_percentage, 1.0)


def output_results(results: Dict) -> None:
    """
    Exportiert Ergebnisse als CSV-Dateien.

    - letter_results.csv (Einzelseiten)
    - letter_summary.csv (Deskriptive Statistik)
    """
    results_df = pd.DataFrame(results).T
    results_df.to_csv("./eval/letter_results.csv", sep=";")

    summary_df = results_df.describe()
    summary_df.to_csv("./eval/letter_summary.csv", sep=";")


def main() -> None:
    """
    Hauptpipeline:

    1. Einlesen der Dateistruktur
    2. Verarbeitung jeder Seite:
        - XML parsen
        - Seitenerkennung
        - Textmetriken berechnen
    3. Speichern der Ergebnisse
    """
    file_dir = Path(r"D:\Briefe-gesamt")
    results = {}

    letter_dic = read_file_structure(file_dir)

    with alive_bar(
        title="Letters processed",
        spinner="twirl",
        total=len(letter_dic.keys()),
        force_tty=True
    ) as bar:

        for hans_id in letter_dic.keys():

            for tif_path, xml_path in zip(
                    letter_dic[hans_id]["tif"],
                    letter_dic[hans_id]["xml"]):

                if not xml_path.name.startswith("._"):

                    xml_root = parse_xml(xml_path)

                    # Seitenerkennung (extern definiert)
                    corner_tup, duration, img, img_dpi = \
                        page_detection(tif_path)

                    page_dict = find_written_space(
                        img, xml_root, corner_tup, img_dpi
                    )

                    if page_dict["num_lines"] > 0:
                        page_dict = calculate_margins(
                            page_dict, img_dpi
                        )

                    results = add_page_results(
                        results, xml_path,
                        page_dict, duration
                    )

            bar()

    output_results(results)


if __name__ == "__main__":
    main()
