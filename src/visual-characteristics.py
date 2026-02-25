from pathlib import Path
from typing import Tuple, List, Dict, Any
import statistics
import math

import cv2
import numpy as np
import pandas as pd
from lxml import etree
import time
from alive_progress import alive_bar
from PIL import Image
from PIL.ExifTags import TAGS
from shapely.geometry import Polygon


def read_file_structure(dir_path: Path) -> dict[str, dict[str, list[Path]]]:
    """globs given directory for lists of paths to xml files as well as tif files"""
    result = {}
    # Durchlaufe alle Ordner B
    for transkribus_folder in dir_path.iterdir():
        for hans_folder in transkribus_folder.iterdir():
            if hans_folder.is_dir():
                images = sorted(
                    (file for file in hans_folder.rglob("*.tif") if file.is_file()),
                    key=lambda f: f.stem
                )
                xml_files = sorted(
                    (file for file in hans_folder.rglob("./page/*.xml") if file.is_file()),
                    key=lambda f: f.stem
                )
                result[hans_folder.name] = {
                    'tif': images,
                    'xml': xml_files
                }
    return result


def parse_xml(file_path: Path) -> etree.Element:
    """parses a single xml file and returns root element tree"""
    try:
        pxml = etree.parse(file_path).getroot()
    except TypeError:
        print(f"xml syntax error in file {file_path}")
        pxml = etree.Element("None")
    return pxml


def page_detection(tif_path) -> tuple[Any, float]:
    """uses page detection and scanning methods to output a rectangle that's assumed to be the page"""
    start_t = time.time()
    # following https://learnopencv.com/automatic-document-scanner-using-opencv/#Getting-Started-with-OpenCV-Document
    # -Scanner with performance optimizations

    # Bild einlesen
    img = cv2.imread(tif_path)
    img_dpi = extract_image_dpi(tif_path)

    # Skalierungsfaktor festlegen (z.B. 0.25 für eine Reduzierung auf 25% der Originalgröße)
    scale_factor = 0.25

    # Bild auf kleinere Auflösung skalieren (behalte das Seitenverhältnis)
    small_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Bildverarbeitung (GrabCut, Kantenerkennung, etc.) auf dem verkleinerten Bild
    kernel = np.ones((3, 3), np.uint8)
    small_img = cv2.morphologyEx(small_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    mask = np.zeros(small_img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, small_img.shape[1] - 20, small_img.shape[0] - 20)

    cv2.grabCut(small_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    small_img = small_img * mask2[:, :, np.newaxis]

    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(page)
    box_points = cv2.boxPoints(rect).astype(int)

    # Koordinaten zurück auf die Originalgröße skalieren
    box_points = (box_points / scale_factor).astype(int)

    # Optional: Zeichnen des erkannten Rechtecks auf dem Originalbild
    cv2.drawContours(img, [box_points], -1, (0, 255, 0), 3)
    for coord in box_points:
        int_coord = tuple(int(coord_val) for coord_val in coord)
        cv2.circle(img, int_coord, 20, (255, 0, 0), -1)
    path_out = Path("./eval/contours/" + str(tif_path.parts[-2]) + "_" + str(tif_path.stem) + "_mask.png")
    cv2.imwrite(path_out, img)
    end_t = time.time()
    duration = end_t - start_t
    return box_points, duration, img, img_dpi


def extract_image_dpi(tif_path):
    with Image.open(tif_path) as img:
        dpi = img.info.get("dpi", (0,0))
        if dpi[0] == dpi[1]:
            return dpi[0]
        else:
            print("ACHTUNG: DPI in horizontaler Richtung ist ungleich DPI in vertikaler Richtung, horizontale DPI werden als Heuristik übernommen.")
            return dpi[0]

def find_written_space(img, xml_root: etree.Element, corner_tup: List[Tuple[float]], img_dpi) -> Tuple[int, int, int, int, int]:
    """finds a rectangle surrounding the written space on a given page and outputs its parameters"""
    page = {}
    page["img_dpi"] = img_dpi
    page["img_w"] = int(xml_root.find("./{*}Page").get("imageWidth"))
    page["img_h"] = int(xml_root.find("./{*}Page").get("imageHeight"))
    page_min_x = np.min(corner_tup[:, 0])  # Minimale x-Koordinate
    page_min_y = np.min(corner_tup[:, 1])  # Minimale y-Koordinate
    page_max_x = np.max(corner_tup[:, 0])  # Maximale x-Koordinate
    page_max_y = np.max(corner_tup[:, 1])  # Maximale y-Koordinate
    page["page_min_x"] = max(0, min(page_min_x,
                                    page["img_w"] - 1))  # Sicherstellen, dass der Wert innerhalb der x-Grenzen liegt
    page["page_max_x"] = max(0, min(page_max_x,
                                    page["img_w"] - 1))  # Sicherstellen, dass der Wert innerhalb der x-Grenzen liegt
    page["page_min_y"] = max(0, min(page_min_y,
                                    page["img_h"] - 1))  # Sicherstellen, dass der Wert innerhalb der y-Grenzen liegt
    page["page_max_y"] = max(0, min(page_max_y,
                                    page["img_h"] - 1))  # Sicherstellen, dass der Wert innerhalb der y-Grenzen liegt
    page["page_w"] = page_max_x - page_min_x
    page["page_h"] = page_max_y - page_min_y
    # Unplausible Ergebnisse der Page Detection vorbeugen, Heuristik 25% des Bildes in jeder Dimension muss von der
    # Seite eingenommen werden
    if page["page_w"] < .25 * page["img_w"]:
        page["page_w"] = page["img_w"]
    if page["page_h"] < .25 * page["img_h"]:
        page["page_h"] = page["img_h"]
    page["page_w_mm"] = pix_to_mm(page["page_w"], img_dpi)
    page["page_h_mm"] = pix_to_mm(page["page_h"], img_dpi)
    # Einheitliche Formate erzeugen: Seiten werden im Hochkant-Format normalisiert
    page["page_longer_mm"] = page["page_h_mm"] if page["page_h_mm"] > page["page_w_mm"] else page["page_w_mm"]
    page["page_shorter_mm"] = page["page_w_mm"] if page["page_w_mm"] < page["page_h_mm"] else page["page_h_mm"]

    page["bgr_color_r"], page["bgr_color_g"], page["bgr_color_b"] = calculate_color(img, page_min_x, page_max_x, page_min_y, page_max_y)

    header = xml_root.find(".//{*}ImageRegion")
    if header is not None:
        header_coords = extract_coords(header)
        print("HEADER: ", header_coords)
        header_x = [x for x, y in header_coords]
        header_y = [y for x, y in header_coords]

        # Bestimme die minimalen und maximalen Werte
        header_x_min, header_x_max = min(header_x), max(header_x)
        header_y_min, header_y_max = min(header_y), max(header_y)
        page["header_x_min"] = header_x_min
        page["header_x_max"] = header_x_max
        page["header_y_min"] = header_y_min
        page["header_y_max"] = header_y_max

        # Berechne die Seitenlängen des Rechtecks
        header_rect_x = header_x_max - header_x_min
        header_rect_y = header_y_max - header_y_min

        page["header_w_mm"] = pix_to_mm(header_rect_x, img_dpi)
        page["header_h_mm"] = pix_to_mm(header_rect_y, img_dpi)

        # Berechne den Flächeninhalt
        page["header_area_mm2"] = page["header_w_mm"] * page["header_h_mm"]
    else:
        page["header_area_mm2"] = 0

    text_regions = {}
    for text_region in xml_root.findall(".//{*}TextRegion"):
        text_region_id = text_region.get("id")
        region_dict = {}
        region_dict["coords_all"] = extract_coords(text_region)
        sorted_x_list = sorted(region_dict["coords_all"], key=lambda t: t[0])
        sorted_y_list = sorted(region_dict["coords_all"], key=lambda t: t[1])
        region_dict["min_x"] = max(0, max(sorted_x_list[0][0], page_min_x))  # Minimale x-Koordinate, begrenzt durch page_min_x
        region_dict["max_x"] = min(sorted_x_list[-1][0], page_max_x)  # Maximale x-Koordinate, begrenzt durch page_max_x
        region_dict["min_y"] = max(0, max(sorted_y_list[0][1], page_min_y))  # Minimale y-Koordinate, begrenzt durch page_min_y
        region_dict["max_y"] = min(sorted_y_list[-1][1], page_max_y)  # Maximale y-Koordinate, begrenzt durch page_max_y
        region_dict["region_h_mm"] = pix_to_mm(region_dict["max_y"] - region_dict["min_y"], img_dpi)
        region_dict["region_w_mm"] = pix_to_mm(region_dict["max_x"] - region_dict["min_x"], img_dpi)
        region_dict["area_mm2"] = region_dict["region_h_mm"] *  region_dict["region_w_mm"]
        try:
            region_dict["area_mm2"] = calculate_polygon_area(region_dict["coords_all"], img_dpi)
        except:
            print(f"ACHTUNG: fehlerhaft erkannte Koordinaten für Region {text_region_id}: {region_dict['coords_all']}")
            region_dict["area_mm2"] = 0
        text_regions[text_region_id] = region_dict

    text_lines = {}
    for text_line in xml_root.findall(".//{*}TextRegion/{*}TextLine"):
        text_lines[text_line.get("id")] = {"coords_all": extract_coords(text_line)}
    for key, value in text_lines.items():
        # Sortieren der Koordinaten nach x (horizontal) und y (vertikal)
        sorted_x_list = sorted(value["coords_all"], key=lambda t: t[0])
        sorted_y_list = sorted(value["coords_all"], key=lambda t: t[1])

        # Berechnung der Extremwerte direkt
        line_min_x = max(0, max(sorted_x_list[0][0], page_min_x))  # Minimale x-Koordinate, begrenzt durch page_min_x
        line_max_x = min(sorted_x_list[-1][0], page_max_x)  # Maximale x-Koordinate, begrenzt durch page_max_x
        line_min_y = max(0, max(sorted_y_list[0][1], page_min_y))  # Minimale y-Koordinate, begrenzt durch page_min_y
        line_max_y = min(sorted_y_list[-1][1], page_max_y)  # Maximale y-Koordinate, begrenzt durch page_max_y

        # Speichern der Extremwerte in der `text_lines`-Datenstruktur
        text_lines[key]["min_x_px"] = line_min_x
        text_lines[key]["max_x_px"] = line_max_x
        text_lines[key]["min_y_px"] = line_min_y
        text_lines[key]["max_y_px"] = line_max_y

        # Berechnung der Höhe und Breite in Millimetern
        text_lines[key]["height_mm"] = pix_to_mm(line_max_y - line_min_y, img_dpi)  # Umrechnung der Höhe
        text_lines[key]["width_mm"] = pix_to_mm(line_max_x - line_min_x, img_dpi)  # Umrechnung der Breite

        # Berechnung des Flächeninhalts des Zeilenpolygons
        text_lines[key]["area_mm2"] = calculate_polygon_area(text_lines[key]["coords_all"], img_dpi)

        # Ränder links und rechts
        text_lines[key]["margin_left_mm"] = pix_to_mm(line_min_x - page_min_x, img_dpi)
        text_lines[key]["margin_right_mm"] = pix_to_mm(page_max_x - line_max_x, img_dpi)

    num_lines = len(text_lines)
    # COMMENT: aggregating measures for single lines on page level
    page["num_lines"] = num_lines

    if num_lines > 0:
        page["text_fill"] = calculate_fill(text_lines, text_regions, page["page_w_mm"], page["page_h_mm"], page["header_area_mm2"])
        page["text_min_x"] = min([text_lines[key]["min_x_px"] for key in text_lines.keys()])
        page["text_max_x"] = max([text_lines[key]["max_x_px"] for key in text_lines.keys()])
        page["text_min_y"] = min([text_lines[key]["min_y_px"] for key in text_lines.keys()])
        page["text_max_y"] = max([text_lines[key]["max_y_px"] for key in text_lines.keys()])
        page["line_height_avg_mm"] = statistics.mean([text_lines[key]["height_mm"] for key in text_lines.keys()])
        page["line_width_avg_mm"] = statistics.mean([text_lines[key]["width_mm"] for key in text_lines.keys()])
        page["line_area_avg_mm2"] = statistics.mean([text_lines[key]["area_mm2"] for key in text_lines.keys()])
        page["line_margin_left_avg_mm"] = statistics.mean(
            [text_lines[key]["margin_left_mm"] for key in text_lines.keys()])
        page["line_margin_right_avg_mm"] = statistics.mean(
            [text_lines[key]["margin_right_mm"] for key in text_lines.keys()])
        page["line_margins_avg_mm"] = (page["line_margin_left_avg_mm"] + page["line_margin_right_avg_mm"]) / 2
        page["line_height_outliers"] = search_for_outliers(text_lines, "height_mm", 2).keys()
        page["line_width_outliers"] = search_for_outliers(text_lines, "width_mm", 2).keys()
        page["margin_left_outliers"] = search_for_outliers(text_lines, "margin_left_mm", 2).keys()
    return page

def extract_coords(xml_element):
    return [tuple(map(int, coord_point.split(","))) for coord_point in xml_element.find("./{*}Coords").get("points").split(" ")]


def get_extreme(val_list: List[Tuple[int]], key: str, i: int, limit: int) -> int:
    """takes a list of sorted tuples, sorted by the tuple value at index i, and outputs the corresponding min/max value.
     Negative values are viewed as implausible as are values that exceed the page size"""
    if val_list:
        val = -1
        if key == "min":
            n = 0
            while val < 0:
                val = val_list[n][i]
                n += 1
        elif key == "max":
            n = -1
            while val < 0:
                val = val_list[n][i]
                n -= 1
        if val > limit:
            val = limit
        return val
    else:
        return 0


def search_for_outliers(data: Dict, key_name: str, num_stdev: float) -> Dict:
    # Umwandlung der Liste in ein NumPy-Array
    values = [item[key_name] for item in data.values()]

    # Schritt 2: Berechne den Mittelwert und die Standardabweichung
    if len(values) < 2:
        return {}  # Keine Outliers, da nur ein Wert

    mean = statistics.mean(values)
    stdev = statistics.stdev(values)

    if stdev == 0:
        return {}  # Keine Outliers, da alle Werte gleich sind

    # Schritt 3: Berechne die Z-Scores und erstelle die Outlier-Maske
    outlier_mask = {
        key: abs((item[key_name] - mean) / stdev) > num_stdev
        for key, item in data.items()
    }
    return outlier_mask


def calculate_margins(page_dict: Dict, img_dpi) -> Dict:
    """calculates margins based on given min / max coordinates of text regions"""
    page_dict["margin_top_mm"] = abs(pix_to_mm(page_dict.get("text_min_y") - page_dict.get("page_min_y"), img_dpi))
    page_dict["margin_bottom_mm"] = abs(pix_to_mm(page_dict.get("page_max_y") - page_dict.get("text_max_y"), img_dpi))
    page_dict["margin_right_mm"] = abs(pix_to_mm(page_dict.get("page_max_x") - page_dict.get("text_max_x"), img_dpi))
    page_dict["margin_left_mm"] = abs(pix_to_mm(page_dict.get("text_min_x") - page_dict.get("page_min_x"), img_dpi))
    if page_dict.get("header_max_y"):
        page_dict["margin_to_header_mm"] = abs(pix_to_mm(page_dict.get("text_min_y") - page_dict.get("header_max_y"), img_dpi))
    # COMMENT: shouldn't I use the average margin instead?
    page_dict["margin_x_percentage"] = (page_dict["margin_left_mm"] + page_dict["margin_right_mm"]) / page_dict.get(
        "page_w")
    page_dict["margin_y_percentage"] = (page_dict["margin_top_mm"] + page_dict["margin_bottom_mm"]) / page_dict.get(
        "page_h")
    return page_dict


def sort_points_by_angle(points):
    # Berechne den Mittelpunkt des Polygons
    centroid_x = sum(x for x, y in points) / len(points)
    centroid_y = sum(y for x, y in points) / len(points)

    def angle_from_centroid(point):
        x, y = point
        return math.atan2(y - centroid_y, x - centroid_x)

    # Sortiere die Punkte nach dem berechneten Winkel
    sorted_points = sorted(points, key=angle_from_centroid)
    return sorted_points


def calculate_polygon_area(coord_list, dpi):
    sorted_coord_list = sort_points_by_angle(coord_list)
    area_px = Polygon(sort_points_by_angle(sorted_coord_list)).area
    area_mm2 = ((25.4 / dpi) ** 2) * area_px
    return area_mm2


def calculate_fill(text_lines, text_regions, page_w_mm, page_h_mm, addition_mm2) -> float:
    """calculates the percentage of the page filled with text lines, text_regions acts as a backup"""
    page_area_mm2 = page_h_mm * page_w_mm
    text_lines_combined_mm2 = sum([float(text_lines[i]["area_mm2"]) for i in text_lines.keys()])
    if page_area_mm2 <= text_lines_combined_mm2:
        text_regions_combined_mm2 = sum([float(text_regions[i]["area_mm2"]) for i in text_regions.keys()])
        area_percentage = (text_regions_combined_mm2 + addition_mm2) / page_area_mm2
    else:
        area_percentage = (text_lines_combined_mm2 + addition_mm2) / page_area_mm2
    return min(area_percentage, 1.0)


def calculate_color(img, x_min, x_max, y_min, y_max) -> Tuple:
    roi = img[y_min:y_max, x_min:x_max]
    mean_color_bgr = cv2.mean(roi)
    return mean_color_bgr[2], mean_color_bgr[1], mean_color_bgr[0]


def add_page_results(results: Dict, xml_path: Path, page_dict, duration) -> Dict:
    page_dict["hans_id"] = str(xml_path.parts[-3])
    page_dict["page"]= str(xml_path.stem)
    results[page_dict["hans_id"] + "_" + page_dict["page"]] = page_dict
    return results


def pix_to_mm(length_pixel: int, dpi: int) -> float:
    """converts length in pixels to length in mm for a given dpi resolution"""
    return (length_pixel * 25.4) / dpi


def output_results(results: Dict) -> None:
    """outputs results as csv files"""
    results_df = pd.DataFrame(results).T
    results_df.to_csv("./eval/letter_results.csv", sep=";")
    summary_df = results_df.describe()
    summary_df.to_csv("./eval/letter_summary.csv", sep=";")


def main() -> None:
    file_dir = Path(r"D:\Briefe-gesamt")
    results = {}
    letter_dic = read_file_structure(file_dir)
    with alive_bar(title="Letters processed", spinner="twirl", total=len(letter_dic.keys()), force_tty=True) as bar:
        for hans_id in letter_dic.keys():
            print(hans_id)
            for tif_path, xml_path in zip(letter_dic[hans_id]["tif"], letter_dic[hans_id]["xml"]):
                if not xml_path.name.startswith("._"):
                    print("|__", xml_path)
                    if not xml_path.stem == tif_path.stem:
                        print("WARNING: File lists are misaligned!")
                        print(xml_path.stem, tif_path.stem)
                    xml_root = parse_xml(xml_path)
                    corner_tup, duration, img, img_dpi = page_detection(tif_path)
                    page_dict = find_written_space(img, xml_root, corner_tup, img_dpi)
                    if page_dict["num_lines"] > 0:
                        page_dict = calculate_margins(page_dict, img_dpi)
                    results = add_page_results(results, xml_path, page_dict, duration)
            bar()
    output_results(results)


if __name__ == "__main__":
    main()
