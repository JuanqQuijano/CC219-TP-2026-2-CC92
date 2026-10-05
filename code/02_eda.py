"""Paso 2: análisis exploratorio de datos (EDA).

Lee el dataset original y genera tablas en data/eda/ y figuras en
informe/figuras/. Si ya existe el dataset procesado en español (paso 3),
también genera las palabras más frecuentes del texto traducido.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config

DetectorFactory.seed = 0
sns.set_theme(style="whitegrid", context="notebook")

STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con",
    "no", "una", "su", "para", "es", "al", "lo", "como", "más", "o", "pero", "sus", "le",
    "ha", "me", "si", "sin", "sobre", "este", "ya", "entre", "cuando", "todo", "esta", "ser",
    "son", "dos", "también", "fue", "había", "era", "muy", "años", "hasta", "desde", "está",
    "mi", "porque", "qué", "sólo", "han", "yo", "hay", "vez", "puede", "todos", "así", "nos",
    "ni", "parte", "tiene", "él", "uno", "donde", "bien", "tiempo", "mismo", "ese", "ahora",
    "cada", "e", "vida", "otro", "después", "te", "otros", "aunque", "esa", "eso", "hace",
    "otra", "gobierno", "tan", "durante", "siempre", "día", "tanto", "ella", "tres", "sí",
    "dijo", "sido", "gran", "país", "según", "menos", "estos", "estas", "les", "nosotros",
    "estamos", "tenemos", "favor", "por favor", "somos", "usted", "ustedes", "quiero",
}


def guardar_figura(nombre: str) -> None:
    plt.tight_layout()
    plt.savefig(config.DIR_FIGURAS / nombre, dpi=150, bbox_inches="tight")
    plt.close()


def balance_preguntas(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in config.ETIQUETAS_PREGUNTAS:
        pos = int(df[col].sum())
        filas.append({
            "etiqueta": col,
            "nombre": config.NOMBRES_ES[col],
            "positivos": pos,
            "negativos": len(df) - pos,
            "pct_positivos": pos / len(df) * 100,
            "ratio_neg_pos": (len(df) - pos) / pos if pos else np.inf,
        })
    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_EDA / "balance_preguntas.csv", index=False)

    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=tabla, x="nombre", y="pct_positivos", color="#3b75af")
    for p, pos in zip(ax.patches, tabla["positivos"]):
        ax.annotate(f"{p.get_height():.1f}%\n(n={pos:,})",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("% de mensajes positivos")
    ax.set_title("Proporción de la clase positiva en las etiquetas de las preguntas")
    ax.set_ylim(0, tabla["pct_positivos"].max() * 1.3)
    guardar_figura("fig01_balance_preguntas.png")
    return tabla


def prevalencia_categorias(df: pd.DataFrame) -> pd.DataFrame:
    etiquetas = [c for c in config.columnas_etiqueta(df) if c != "related"]
    tabla = (df[etiquetas].mean() * 100).sort_values(ascending=False).rename("pct_positivos")
    tabla.to_frame().to_csv(config.DIR_EDA / "prevalencia_categorias.csv")

    colores = ["#d1495b" if c in config.ETIQUETAS_PREGUNTAS else "#9bb7d4" for c in tabla.index]
    plt.figure(figsize=(8, 9))
    plt.barh(tabla.index[::-1], tabla.values[::-1], color=colores[::-1])
    plt.xlabel("% de mensajes positivos")
    plt.title("Prevalencia de las 35 categorías binarias\n(en rojo, las usadas en las preguntas)")
    guardar_figura("fig02_prevalencia_categorias.png")
    return tabla


def genero_y_split(df: pd.DataFrame) -> pd.DataFrame:
    tabla = pd.crosstab(df["genre"], df["split"])[["train", "validation", "test"]]
    tabla.to_csv(config.DIR_EDA / "genero_por_split.csv")

    tasas = df.groupby("genre")[config.ETIQUETAS_PREGUNTAS].mean() * 100
    tasas.to_csv(config.DIR_EDA / "tasa_positivos_por_genero.csv")

    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))
    tabla.plot(kind="bar", ax=ejes[0], rot=0, color=["#3b75af", "#f2a541", "#6aa84f"])
    ejes[0].set_title("Mensajes por género y partición")
    ejes[0].set_xlabel("Género")
    ejes[0].set_ylabel("Número de mensajes")

    tasas.rename(columns=config.NOMBRES_ES).T.plot(kind="bar", ax=ejes[1], rot=20)
    ejes[1].set_title("% de positivos por género")
    ejes[1].set_ylabel("% de mensajes")
    ejes[1].legend(title="Género")
    guardar_figura("fig03_genero_split.png")
    return tabla


def longitud_mensajes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(n_palabras=df["message"].str.split().str.len(),
                   n_caracteres=df["message"].str.len())
    tabla = df.groupby("genre")[["n_palabras", "n_caracteres"]].describe().round(1)
    tabla.to_csv(config.DIR_EDA / "longitud_por_genero.csv")

    plt.figure(figsize=(9, 4.5))
    sns.histplot(data=df, x="n_palabras", hue="genre", log_scale=True,
                 element="step", stat="density", common_norm=False, bins=50)
    plt.xlabel("Palabras por mensaje (escala log)")
    plt.ylabel("Densidad")
    plt.title("Distribución de la longitud de los mensajes por género")
    guardar_figura("fig04_longitud_mensajes.png")
    return tabla


def correlacion_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["related", "aid_related", "direct_report", *config.ETIQUETAS_PREGUNTAS,
            "other_aid", "refugees", "death"]
    datos = df[cols].assign(related=(df["related"] == 1).astype(int))
    corr = datos.corr()
    corr.to_csv(config.DIR_EDA / "correlacion_etiquetas.csv")

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot_kws={"fontsize": 8})
    plt.title("Correlación (phi) entre etiquetas seleccionadas")
    guardar_figura("fig05_correlacion_etiquetas.png")
    return corr


def etiquetas_por_mensaje(df: pd.DataFrame) -> pd.Series:
    etiquetas = [c for c in config.columnas_etiqueta(df) if c != "related"]
    n = df[etiquetas].sum(axis=1)
    conteo = n.value_counts().sort_index()
    conteo.rename("mensajes").to_frame().to_csv(config.DIR_EDA / "etiquetas_por_mensaje.csv")

    tipos = df[config.ETIQUETAS_P3].sum(axis=1)
    tipos.value_counts().sort_index().rename("mensajes").to_frame().to_csv(
        config.DIR_EDA / "tipos_ayuda_por_mensaje.csv")

    plt.figure(figsize=(9, 4))
    plt.bar(conteo.index, conteo.values, color="#3b75af")
    plt.xlabel("Número de categorías positivas (sin contar `related`)")
    plt.ylabel("Mensajes")
    plt.title("Cardinalidad de etiquetas por mensaje")
    guardar_figura("fig06_etiquetas_por_mensaje.png")
    return conteo


def detectar_idioma(texto) -> str:
    if not isinstance(texto, str) or len(texto.strip()) < 3:
        return "vacio"
    try:
        return detect(texto)
    except LangDetectException:
        return "indeterminado"


def idioma_original(df: pd.DataFrame) -> pd.DataFrame:
    if config.CSV_IDIOMA.exists():
        idiomas = pd.read_csv(config.CSV_IDIOMA)["idioma_original"]
    else:
        print("Detectando idioma del campo `original` (puede tardar unos minutos)...")
        idiomas = df["original"].map(detectar_idioma)
        pd.DataFrame({"idioma_original": idiomas}).to_csv(config.CSV_IDIOMA, index=False)

    df = df.assign(idioma_original=idiomas.values)
    tabla = pd.crosstab(df["idioma_original"], df["genre"])
    tabla["total"] = tabla.sum(axis=1)
    tabla = tabla.sort_values("total", ascending=False)
    tabla.to_csv(config.DIR_EDA / "idioma_original_por_genero.csv")

    top = tabla.drop(index="vacio", errors="ignore").head(10)
    plt.figure(figsize=(9, 4))
    top.drop(columns="total").plot(kind="bar", stacked=True, rot=0, ax=plt.gca())
    plt.xlabel("Idioma detectado (código ISO 639-1)")
    plt.ylabel("Mensajes")
    plt.title("Idioma detectado por langdetect en el campo `original` (excluye vacíos)\n"
              "langdetect no soporta criollo haitiano: fr, id, tl, af, sl... son mayormente criollo")
    guardar_figura("fig07_idioma_original.png")
    return tabla


def palabras_frecuentes(textos: pd.Series, stopwords: set[str], n: int = 15) -> pd.Series:
    tokens = textos.str.lower().str.findall(r"[a-záéíóúñü]{3,}").explode()
    tokens = tokens[~tokens.isin(stopwords)]
    return tokens.value_counts().head(n)


def figura_palabras(df: pd.DataFrame, col_texto: str, stopwords: set[str],
                    nombre: str, idioma: str) -> pd.DataFrame:
    grupos = {
        "Petición (request=1)": df.loc[df["request"] == 1, col_texto],
        "Oferta (offer=1)": df.loc[df["offer"] == 1, col_texto],
        "Sin petición ni oferta": df.loc[(df["request"] == 0) & (df["offer"] == 0), col_texto],
    }
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    filas = []
    for ax, (titulo, textos) in zip(ejes, grupos.items()):
        top = palabras_frecuentes(textos.dropna(), stopwords)
        ax.barh(top.index[::-1], top.values[::-1], color="#3b75af")
        ax.set_title(titulo)
        ax.set_xlabel("Frecuencia")
        filas += [{"grupo": titulo, "palabra": p, "frecuencia": f} for p, f in top.items()]
    fig.suptitle(f"Palabras más frecuentes por clase ({idioma}, sin stopwords)")
    guardar_figura(nombre)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(config.DIR_EDA / nombre.replace(".png", ".csv"), index=False)
    return tabla


def revisar_consistencia(df: pd.DataFrame) -> dict:
    etiquetas = [c for c in config.columnas_etiqueta(df) if c != "related"]
    no_relacionados = df["related"] != 1
    resumen = {
        "related_0": int((df["related"] == 0).sum()),
        "related_2": int((df["related"] == 2).sum()),
        "no_relacionados_con_alguna_etiqueta": int(df.loc[no_relacionados, etiquetas].any(axis=1).sum()),
        "duplicados_message": int(df.duplicated(subset="message").sum()),
        "duplicados_message_entre_splits": int(
            df.groupby("message")["split"].nunique().gt(1).sum()),
        "request_y_offer_a_la_vez": int(((df["request"] == 1) & (df["offer"] == 1)).sum()),
        "offer_positivos_por_split": df.groupby("split")["offer"].sum().to_dict(),
    }
    pd.Series(resumen).to_csv(config.DIR_EDA / "consistencia.csv", header=["valor"])
    return resumen


def main() -> None:
    config.crear_directorios()
    df = pd.read_csv(config.CSV_ORIGINAL)

    print("Balance de las preguntas:\n", balance_preguntas(df).round(2).to_string(index=False))
    print("\nPrevalencia (top 10):\n", prevalencia_categorias(df).head(10).round(2).to_string())
    print("\nGénero por split:\n", genero_y_split(df).to_string())
    print("\nLongitud por género (palabras):\n",
          longitud_mensajes(df)["n_palabras"].to_string())
    correlacion_etiquetas(df)
    print("\nEtiquetas por mensaje:\n", etiquetas_por_mensaje(df).to_string())
    print("\nIdioma de `original`:\n", idioma_original(df).head(12).to_string())
    print("\nConsistencia:", revisar_consistencia(df))
    figura_palabras(df, "message", set(ENGLISH_STOP_WORDS) | {"http", "www", "com"},
                    "fig08_palabras_frecuentes_en.png", "inglés")

    if config.CSV_PROCESADO.exists():
        dfp = pd.read_csv(config.CSV_PROCESADO)
        dfp = dfp[dfp["message_es"].notna()]
        figura_palabras(dfp, "message_es", STOPWORDS_ES, "fig09_palabras_frecuentes_es.png", "español")
        dfp = dfp.assign(n_es=dfp["message_es"].str.split().str.len(),
                         n_en=dfp["message"].str.split().str.len())
        resumen = dfp[["n_en", "n_es"]].describe().round(1)
        resumen.to_csv(config.DIR_EDA / "longitud_en_vs_es.csv")
        print("\nLongitud inglés vs. español (palabras):\n", resumen.to_string())

    print(f"\nTablas en {config.DIR_EDA}\nFiguras en {config.DIR_FIGURAS}")


if __name__ == "__main__":
    main()
