"""Paso 4: validación de la traducción automática por retrotraducción.

Toma una muestra estratificada por género de los mensajes traducidos, los
vuelve a traducir al inglés con OPUS-MT (es-en) y compara con el `message`
original mediante chrF y BLEU. Guarda la muestra para revisión manual.
"""

import argparse

import pandas as pd
import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

import config

MODELO_RETRO = "Helsinki-NLP/opus-mt-es-en"


def retrotraducir(textos: list[str], tam_lote: int = 32) -> list[str]:
    tokenizer = MarianTokenizer.from_pretrained(MODELO_RETRO)
    modelo = MarianMTModel.from_pretrained(MODELO_RETRO).eval()
    modelo.generation_config.max_length = None
    salida = []
    for i in range(0, len(textos), tam_lote):
        entradas = tokenizer(textos[i:i + tam_lote], return_tensors="pt", padding=True,
                             truncation=True, max_length=256)
        with torch.inference_mode():
            gen = modelo.generate(**entradas, num_beams=1, max_new_tokens=256)
        salida += tokenizer.batch_decode(gen, skip_special_tokens=True)
    return salida


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-por-genero", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(config.CSV_PROCESADO)
    df = df[df["fuente_es"] == "opus-mt-en-es"]
    muestra = df.groupby("genre").sample(n=args.n_por_genero, random_state=config.SEMILLA)
    muestra = muestra.assign(retro_en=retrotraducir(muestra["message_es"].tolist()))

    filas = []
    for genero, g in [("total", muestra), *muestra.groupby("genre")]:
        refs = [g["message"].tolist()]
        filas.append({
            "genero": genero,
            "n": len(g),
            "chrF": sacrebleu.corpus_chrf(g["retro_en"].tolist(), refs).score,
            "BLEU": sacrebleu.corpus_bleu(g["retro_en"].tolist(), refs).score,
            "ratio_longitud_es_en": (g["message_es"].str.split().str.len()
                                     / g["message"].str.split().str.len()).median(),
        })
    resumen = pd.DataFrame(filas).round(2)
    resumen.to_csv(config.DIR_EDA / "retrotraduccion_resumen.csv", index=False)
    muestra[["id", "genre", "message", "message_es", "retro_en"]].to_csv(
        config.DIR_EDA / "retrotraduccion_muestra.csv", index=False, encoding="utf-8")
    print(resumen.to_string(index=False))


if __name__ == "__main__":
    main()
