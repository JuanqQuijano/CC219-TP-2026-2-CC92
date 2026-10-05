"""Paso 1: carga del dataset desde Hugging Face e inspección inicial.

Descarga los tres splits de Disaster Response Messages, los une en un único
DataFrame (la columna `split` conserva la partición original) y guarda el
dataset original en data/original/. Imprime un resumen de la estructura.
"""

import pandas as pd
from datasets import load_dataset

import config


def cargar_dataset() -> pd.DataFrame:
    ds = load_dataset(config.DATASET_HF)
    partes = [ds[split].to_pandas() for split in ("train", "validation", "test")]
    return pd.concat(partes, ignore_index=True)


def inspeccionar(df: pd.DataFrame) -> None:
    etiquetas = config.columnas_etiqueta(df)

    print("=" * 70)
    print(f"Forma del dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Columnas de texto: {config.COLUMNAS_TEXTO}")
    print(f"Columnas de etiqueta ({len(etiquetas)}): {etiquetas}")

    print("\nMensajes por split:")
    print(df["split"].value_counts().to_string())

    print("\nMensajes por género:")
    print(df["genre"].value_counts().to_string())

    print("\nTipos de datos:")
    print(df.dtypes.value_counts().to_string())

    print("\nValores nulos o vacíos por columna de texto:")
    for col in config.COLUMNAS_TEXTO:
        vacios = df[col].isna() | (df[col].astype(str).str.strip() == "")
        print(f"  {col:<10} {vacios.sum():>6,} ({vacios.mean():.1%})")

    print("\nValores de `related` (0=no, 1=sí, 2=quizá):")
    print(df["related"].value_counts().sort_index().to_string())

    for col in (config.ETIQUETA_P1, config.ETIQUETA_P2):
        vc = df[col].value_counts().sort_index()
        print(f"\nValores de `{col}`:")
        print(vc.to_string())
        print(f"  Proporción positiva: {df[col].mean():.2%}")

    fuera_binario = [c for c in etiquetas if c != "related" and not df[c].isin([0, 1]).all()]
    print(f"\nEtiquetas no binarias además de `related`: {fuera_binario or 'ninguna'}")

    sin_positivos = [c for c in etiquetas if df[c].sum() == 0]
    print(f"Etiquetas sin ningún positivo: {sin_positivos or 'ninguna'}")

    dup = df.duplicated(subset="message").sum()
    print(f"Mensajes con texto duplicado (`message`): {dup:,}")

    print("\nEjemplos:")
    for _, fila in df.sample(3, random_state=config.SEMILLA).iterrows():
        print(f"  [{fila['genre']}] request={fila['request']} offer={fila['offer']}")
        print(f"    message : {fila['message'][:150]}")
        print(f"    original: {str(fila['original'])[:150]}")
    print("=" * 70)


def main() -> None:
    config.crear_directorios()
    df = cargar_dataset()
    df.to_csv(config.CSV_ORIGINAL, index=False, encoding="utf-8")
    print(f"Dataset original guardado en {config.CSV_ORIGINAL}")
    inspeccionar(df)


if __name__ == "__main__":
    main()
