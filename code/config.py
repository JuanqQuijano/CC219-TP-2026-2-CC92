"""Rutas y constantes compartidas por los scripts del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DIR_DATA = RAIZ / "data"
DIR_ORIGINAL = DIR_DATA / "original"
DIR_EDA = DIR_DATA / "eda"
DIR_PROCESADO = DIR_DATA / "procesado"
DIR_FIGURAS = RAIZ / "informe" / "figuras"

CSV_ORIGINAL = DIR_ORIGINAL / "disaster_response_messages.csv"
CSV_IDIOMA = DIR_EDA / "idioma_original.csv"
CSV_TRADUCCIONES = DIR_PROCESADO / "traducciones_checkpoint.csv"
CSV_RETRADUCCIONES = DIR_PROCESADO / "retraducciones_por_oracion.csv"
CSV_PROCESADO = DIR_PROCESADO / "disaster_response_messages_es.csv"

DATASET_HF = "community-datasets/disaster_response_messages"
MODELO_TRADUCCION = "Helsinki-NLP/opus-mt-en-es"
SEMILLA = 42

COLUMNAS_TEXTO = ["split", "message", "original", "genre"]

# Etiquetas de las preguntas de clasificación del caso de uso.
ETIQUETA_P1 = "request"
ETIQUETA_P2 = "offer"
ETIQUETAS_P3 = ["water", "food", "shelter", "medical_help"]
ETIQUETAS_PREGUNTAS = [ETIQUETA_P1, ETIQUETA_P2, *ETIQUETAS_P3]

NOMBRES_ES = {
    "request": "Petición",
    "offer": "Oferta",
    "water": "Agua",
    "food": "Alimentos",
    "shelter": "Refugio",
    "medical_help": "Ayuda médica",
}


def crear_directorios() -> None:
    for d in (DIR_ORIGINAL, DIR_EDA, DIR_PROCESADO, DIR_FIGURAS):
        d.mkdir(parents=True, exist_ok=True)


def columnas_etiqueta(df) -> list[str]:
    return [c for c in df.columns if c not in COLUMNAS_TEXTO]
