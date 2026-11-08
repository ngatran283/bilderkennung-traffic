# Traffic Sign Recognition (GTSRB)

Deep-Learning-Modell zur Erkennung und Klassifizierung von Verkehrszeichen.
Datensatz: German Traffic Sign Recognition Benchmark — 43 Klassen, ~50.000 Bilder.

---

## Setup

### 1. Repository klonen

```bash
git clone <repo-url>
cd traffic-sign-recognition
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3. Datensatz herunterladen

Der Datensatz ist **nicht** im Repository enthalten (zu groß).
Manuell herunterladen von Kaggle:

👉 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Nach dem Download die Ordnerstruktur wie folgt anlegen:

```
traffic-sign-recognition/
└── data/
    ├── Train/
    │   ├── 0/
    │   ├── 1/
    │   └── ... (43 Klassen)
    ├── Test/
    │   ├── 00000.png
    │   └── ...
    ├── Train.csv
    ├── Test.csv
    └── Meta.csv
```

---

## Projektstruktur

```
traffic-sign-recognition/
├── data/               # Datensatz (nicht im Git)
├── notebooks/
│   └── 1-EDA.ipynb     # Explorative Datenanalyse
├── src/
│   └── data/
│       └── data_loader.py
├── requirements.txt
└── README.md
```

---

## Ausführen

### EDA Notebook starten

```bash
jupyter notebook notebooks/1-EDA.ipynb
```