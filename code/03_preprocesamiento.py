"""Paso 3: limpieza, construcción del texto en español y normalización.

1. Elimina mensajes vacíos y duplicados exactos de `message`.
2. Construye `message_es`: si el campo `original` está en español se usa
   directamente; en caso contrario se traduce `message` (inglés) con OPUS-MT.
3. Detecta traducciones con omisiones, truncadas o degeneradas y las vuelve a
   traducir oración por oración con beam search.
4. Normaliza el texto (`message_es_norm`, `message_norm`), elimina los mensajes
   sin texto útil y añade una partición estratificada por (request, offer).
5. Guarda el dataset preparado en data/procesado/.

La traducción se hace por lotes y se guarda en un checkpoint, de modo que el
script puede interrumpirse y reanudarse sin perder lo ya traducido.

Uso:
    python 03_preprocesamiento.py                 # traduce todo lo pendiente
    python 03_preprocesamiento.py --limite 2000   # traduce como máximo 2000 mensajes
    python 03_preprocesamiento.py --sin-traducir  # solo regenera el CSV final
"""

import argparse
import re
import time
import unicodedata

import pandas as pd

import config

PALABRAS_FUNCION_ES = {
    "de", "la", "que", "el", "en", "los", "las", "por", "para", "con", "una", "es", "del",
    "nos", "nuestro", "estamos", "hay", "ayuda", "favor", "situacion", "agua", "gracias",
    "necesitamos",
}

RE_URL = re.compile(r"(https?://\S+|www\.\S+|\bhttp\b)", re.IGNORECASE)
RE_MENCION = re.compile(r"@\w+")
RE_ESPACIOS = re.compile(r"\s+")
RE_REPETIDOS = re.compile(r"([!?.,¡¿'#*])\1+")
RE_RACHA = re.compile(r"(\S)\1{9,}")
RE_ORACIONES = re.compile(r"(?<=[.!?;])\s+")


def limpiar_para_traducir(texto: str) -> str:
    """La puntuación repetida (p. ej. "!!!!!!") hace que Marian entre en bucles."""
    texto = RE_URL.sub(" ", texto)
    texto = RE_MENCION.sub(" ", texto)
    texto = RE_REPETIDOS.sub(r"\1", texto)
    return RE_ESPACIOS.sub(" ", texto).strip()


def normalizar(texto) -> str:
    """Normalización ligera: conserva tildes y ñ porque son informativas en español."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFC", texto).lower()
    texto = RE_URL.sub(" ", texto)
    texto = RE_MENCION.sub(" ", texto)
    texto = texto.replace("#", " ")
    texto = RE_REPETIDOS.sub(r"\1", texto)
    texto = re.sub(r"[^\w\sáéíóúñü¿?¡!.,]", " ", texto)
    return RE_ESPACIOS.sub(" ", texto).strip()


def es_espanol_confiable(texto, idioma: str) -> bool:
    """langdetect confunde el criollo haitiano con español; se exige además
    probabilidad alta y presencia de palabras funcionales del español."""
    if idioma != "es" or not isinstance(texto, str):
        return False
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    mejor = max(detect_langs(texto), key=lambda l: l.prob)
    palabras = set(re.findall(r"\w+", texto.lower()))
    return mejor.lang == "es" and mejor.prob >= 0.95 and len(palabras & PALABRAS_FUNCION_ES) >= 2


def particion_estratificada(df: pd.DataFrame) -> pd.Series:
    """Partición 80/10/10 estratificada por (request, offer).

    En la versión de Hugging Face todos los positivos de `offer` están en test,
    por lo que la partición original no permite entrenar la pregunta 2.
    """
    from sklearn.model_selection import train_test_split

    estrato = df["request"].astype(str) + "_" + df["offer"].astype(str)
    idx_train, idx_resto = train_test_split(df.index, test_size=0.2, stratify=estrato,
                                            random_state=config.SEMILLA)
    idx_val, idx_test = train_test_split(idx_resto, test_size=0.5, stratify=estrato[idx_resto],
                                         random_state=config.SEMILLA)
    split = pd.Series("train", index=df.index)
    split[idx_val] = "validation"
    split[idx_test] = "test"
    return split


def cargar_y_limpiar() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(config.CSV_ORIGINAL)
    df.insert(0, "id", range(len(df)))
    if not config.CSV_IDIOMA.exists():
        raise SystemExit("Falta data/eda/idioma_original.csv: ejecuta antes 02_eda.py")
    df["idioma_original"] = pd.read_csv(config.CSV_IDIOMA)["idioma_original"].values

    registro = {"filas_iniciales": len(df)}
    df = df[df["message"].notna() & (df["message"].str.strip() != "")]
    registro["tras_quitar_vacios"] = len(df)
    df = df.drop_duplicates(subset="message", keep="first").reset_index(drop=True)
    registro["tras_quitar_duplicados"] = len(df)

    df["original_es"] = [es_espanol_confiable(t, i)
                         for t, i in zip(df["original"], df["idioma_original"])]
    registro["original_en_espanol"] = int(df["original_es"].sum())
    df["split_estratificado"] = particion_estratificada(df)
    return df, registro


def cargar_checkpoint() -> pd.DataFrame:
    if config.CSV_TRADUCCIONES.exists():
        return pd.read_csv(config.CSV_TRADUCCIONES)
    return pd.DataFrame(columns=["id", "message_es"])


def cargar_traductor():
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(config.MODELO_TRADUCCION)
    modelo = MarianMTModel.from_pretrained(config.MODELO_TRADUCCION).eval()
    modelo.generation_config.max_length = None
    return tokenizer, modelo


def traducir(textos: list[str], tokenizer, modelo, max_tokens: int, num_beams: int = 1) -> list[str]:
    import torch

    entradas = tokenizer(textos, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_tokens)
    with torch.inference_mode():
        salida = modelo.generate(**entradas, num_beams=num_beams, max_new_tokens=max_tokens)
    return tokenizer.batch_decode(salida, skip_special_tokens=True)


def traducir_pendientes(pendientes: pd.DataFrame, tam_lote: int, max_tokens: int) -> None:
    tokenizer, modelo = cargar_traductor()

    # Ordenar por longitud reduce el relleno dentro de cada lote.
    pendientes = pendientes.assign(_len=pendientes["message"].str.len()).sort_values("_len")
    escribir_cabecera = not config.CSV_TRADUCCIONES.exists()
    total, inicio = len(pendientes), time.time()

    for i in range(0, total, tam_lote):
        lote = pendientes.iloc[i:i + tam_lote]
        textos = [limpiar_para_traducir(t) or "." for t in lote["message"]]
        traducciones = traducir(textos, tokenizer, modelo, max_tokens)

        pd.DataFrame({"id": lote["id"].values, "message_es": traducciones}).to_csv(
            config.CSV_TRADUCCIONES, mode="a", header=escribir_cabecera,
            index=False, encoding="utf-8")
        escribir_cabecera = False

        hechos = min(i + tam_lote, total)
        transcurrido = time.time() - inicio
        restante = transcurrido / hechos * (total - hechos)
        print(f"  {hechos:>6,}/{total:,} traducidos | {transcurrido / 60:5.1f} min "
              f"| restante estimado {restante / 60:5.1f} min", flush=True)


def colapsar_rachas(texto) -> str:
    """Marian a veces entra en bucle y repite un signo cientos de veces."""
    if not isinstance(texto, str):
        return ""
    return RE_RACHA.sub(r"\1", texto)


def es_degenerada(texto) -> bool:
    """Sin contenido alfabético útil. Las rachas de puntuación no cuentan."""
    return sum(c.isalpha() for c in normalizar(colapsar_rachas(texto))) < 3


def detectar_problematicas(df: pd.DataFrame, traducciones: pd.DataFrame) -> pd.DataFrame:
    """Traducciones con omisiones (menos de la mitad de palabras), truncadas
    (mensajes muy largos) o degeneradas (repeticiones sin letras)."""
    t = df[["id", "message"]].merge(traducciones.drop_duplicates("id"), on="id")
    n_en = t["message"].str.split().str.len()
    n_es = t["message_es"].fillna("").str.split().str.len()
    t["motivo"] = None
    t.loc[(n_es < 0.5 * n_en) & (n_en >= 5), "motivo"] = "omision"
    t.loc[n_en > 150, "motivo"] = "truncada"
    t.loc[t["message_es"].map(es_degenerada), "motivo"] = "degenerada"
    return t[t["motivo"].notna()]


def retraducir_por_oraciones(problematicas: pd.DataFrame, max_tokens: int) -> pd.DataFrame:
    """Marian con decodificación voraz tiende a omitir oraciones en mensajes
    con varias de ellas; traducirlas por separado con beam search lo corrige."""
    tokenizer, modelo = cargar_traductor()
    filas = []
    for i, (id_, mensaje, motivo) in enumerate(
            problematicas[["id", "message", "motivo"]].itertuples(index=False), 1):
        texto = limpiar_para_traducir(mensaje)
        oraciones = [o for o in RE_ORACIONES.split(texto) if re.search(r"[A-Za-z]", o)]
        traduccion = " ".join(traducir(oraciones, tokenizer, modelo, max_tokens, num_beams=4)) \
            if oraciones else ""
        filas.append({"id": id_, "motivo": motivo, "message_es": traduccion})
        if i % 50 == 0:
            print(f"  {i}/{len(problematicas)} retraducidos", flush=True)
    resultado = pd.DataFrame(filas)
    resultado.to_csv(config.CSV_RETRADUCCIONES, index=False, encoding="utf-8")
    return resultado


def construir_final(df: pd.DataFrame, traducciones: pd.DataFrame,
                    retraducciones: pd.DataFrame, registro: dict) -> pd.DataFrame:
    es_espanol = df["original_es"]
    df = df.merge(traducciones.drop_duplicates("id"), on="id", how="left")
    df["fuente_es"] = "pendiente"
    df.loc[df["message_es"].notna(), "fuente_es"] = "opus-mt-en-es"

    df["message_es"] = df["message_es"].map(colapsar_rachas)
    if len(retraducciones):
        # La segunda pasada solo se conserva si aporta texto útil. En los casos
        # degenerados el beam search empeoró el bucle de puntuación.
        utiles = retraducciones[~retraducciones["message_es"].map(es_degenerada)].copy()
        nuevas = df["id"].map(utiles.set_index("id")["message_es"].map(colapsar_rachas))
        df.loc[nuevas.notna(), "message_es"] = nuevas[nuevas.notna()]
        df.loc[nuevas.notna(), "fuente_es"] = "opus-mt-en-es-por-oracion"

    df.loc[es_espanol, "message_es"] = df.loc[es_espanol, "original"]
    df.loc[es_espanol, "fuente_es"] = "original_es"

    df["message_norm"] = df["message"].map(normalizar)
    df["message_es_norm"] = df["message_es"].map(normalizar)

    traducido = df["fuente_es"] != "pendiente"
    vacio = traducido & df["message_es"].map(es_degenerada)
    registro["eliminados_texto_es_vacio"] = int(vacio.sum())
    df = df[~vacio]
    registro["filas_finales"] = len(df)

    columnas = ["id", "split", "split_estratificado", "genre", "message", "original", "idioma_original",
                "fuente_es", "message_es", "message_norm", "message_es_norm",
                "related", "aid_related", *config.ETIQUETAS_PREGUNTAS]
    return df[columnas]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite", type=int, default=None,
                        help="máximo de mensajes a traducir en esta ejecución")
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--sin-traducir", action="store_true")
    args = parser.parse_args()

    config.crear_directorios()
    df, registro = cargar_y_limpiar()
    print("Limpieza:", registro)

    traducidos = set(cargar_checkpoint()["id"])
    pendientes = df[~df["original_es"] & ~df["id"].isin(traducidos)]
    print(f"Mensajes con `original` en español (sin traducir): {df['original_es'].sum():,}")
    print(f"Ya traducidos en checkpoint: {len(traducidos):,} | pendientes: {len(pendientes):,}")

    if not args.sin_traducir and len(pendientes):
        if args.limite:
            pendientes = pendientes.sample(min(args.limite, len(pendientes)),
                                           random_state=config.SEMILLA)
        print(f"Traduciendo {len(pendientes):,} mensajes con {config.MODELO_TRADUCCION}...")
        traducir_pendientes(pendientes, args.lote, args.max_tokens)

    traducciones = cargar_checkpoint()
    retraducciones = pd.DataFrame(columns=["id", "motivo", "message_es"])
    if config.CSV_RETRADUCCIONES.exists():
        retraducciones = pd.read_csv(config.CSV_RETRADUCCIONES)
    elif not args.sin_traducir and traducciones["id"].nunique() >= (~df["original_es"]).sum():
        problematicas = detectar_problematicas(df, traducciones)
        print(f"\nTraducciones problemáticas: {problematicas['motivo'].value_counts().to_dict()}")
        print("Retraduciendo por oraciones con beam search...")
        retraducciones = retraducir_por_oraciones(problematicas, args.max_tokens)
    if len(retraducciones):
        registro["retraducidos_por_motivo"] = retraducciones["motivo"].value_counts().to_dict()

    final = construir_final(df, traducciones, retraducciones, registro)
    print("\nRegistro de preprocesamiento:", registro)
    pd.Series(registro).to_csv(config.DIR_EDA / "registro_preprocesamiento.csv", header=["valor"])
    final.to_csv(config.CSV_PROCESADO, index=False, encoding="utf-8")
    print("\nFuente del texto en español:")
    print(final["fuente_es"].value_counts().to_string())
    print(f"\nDataset procesado guardado en {config.CSV_PROCESADO} ({len(final):,} filas)")


if __name__ == "__main__":
    main()
