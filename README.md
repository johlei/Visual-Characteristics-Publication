# Grundlegendes

Dieses Repository enthält Code und Visualisierungen, die die Grundlage für folgenden Aufsatz bilden: J. Nantke, J. Leitgeb & C. Reul. (2026). 'Quantitative Analyse visueller Charakteristika von Briefen'. <i>Zeitschrift für digitale Geisteswissenschaften</i> \[in Vorbereitung\].

Im Unterordner `src` ist der Python-Code zu finden. Die Visualisierungen befinden sich im Unterordner `docs`.

Alternativ können die (interaktiven) Visualisierungen direkt im Browser betrachtet werden: <a href="https://johlei.github.io/Visual-Characteristics-Publication/">Link</a>. 

# Dokumentation

Dieses Projekt analysiert gescannte historische Briefe (TIFF) in Kombination mit PAGE-XML (Transkribus) und extrahiert quantitative visuelle Merkmale der Dokumente.

## Funktionen

### 1. Feature-Extraktion (`visual-characteristics.py`)

- Seitengröße (mm)
- Textflächenanteil
- Zeilenmetriken (Höhe, Breite, Fläche)
- Seitenränder
- Header-Fläche
- Papierfarbe (RGB)
- Ausreißer-Erkennung (Z-Score)

Ergebnisse:
- `eval/letter_results.csv`
- `eval/letter_summary.csv`

---

### 2. Visualisierung (`generate-visualisation.py`)

Erstellt interaktive Plotly-Visualisierungen:

- Seitenformate & Proportionen
- Historische Buchformate (Overlay)
- Textflächenanteil
- Papierfarbe im 3D-RGB-Raum
- Briefe pro Jahr

Ausgabe: HTML-Dateien in `visualizations/`

---

## Erwartete Datenstruktur

```
BaseFolder/
 └── TranskribusExport/
      └── Hans_ID/
           ├── *.tif
           └── page/*.xml
```

## Abhängigkeiten

- OpenCV
- NumPy
- Pandas
- Plotly
- lxml
- Shapely
- Pillow
- alive-progress
