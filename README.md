# Clasificación de peticiones y ofertas de ayuda en mensajes de crisis

Trabajo parcial y final del curso **1ACC0219 Aplicaciones de Data Science** (CC219), sección 4896, ciclo 2026-2. Universidad Peruana de Ciencias Aplicadas.

Repositorio: [JuanqQuijano/CC219-TP-2026-2-CC92](https://github.com/JuanqQuijano/CC219-TP-2026-2-CC92)

## Objetivo

Evaluar clasificadores compactos y de bajo costo que, a partir de mensajes de desastres traducidos al español, respondan cuatro preguntas:

1. ¿El mensaje contiene una petición de ayuda? (`request`)
2. ¿El mensaje contiene una oferta de ayuda? (`offer`)
3. ¿Qué tipo de ayuda se menciona: agua, alimentos, refugio o ayuda médica? (`water`, `food`, `shelter`, `medical_help`)
4. ¿El mensaje está relacionado con un desastre? (`related`), como filtro previo

Este primer hito cubre la carga, el análisis exploratorio, la normalización del texto y la propuesta de modelos. El entrenamiento queda para el trabajo final.

## Integrantes

| Código | Apellidos y nombres |
| :-- | :-- |
| u20231b249 | Almeida Aguilar, Ivan Antonio |
| u202221068 | Quijano Calderon, Juan Miguel |
| u202320608 | Toledo Mamani, Wilber Franz |

Docente: Lopez Abarca, Gustavo Adolfo.

## Dataset

Se usa [Disaster Response Messages](https://huggingface.co/datasets/community-datasets/disaster_response_messages) (Appen / Figure Eight), publicado en Hugging Face por community-datasets. Reúne mensajes del terremoto de Haití (2010), el de Chile (2010), las inundaciones de Pakistán (2010), el huracán Sandy (2012) y noticias de cientos de desastres.

| Archivo | Contenido |
| :-- | :-- |
| `data/original/disaster_response_messages.csv` | Corpus original: 26 248 mensajes, texto en inglés (`message`), texto de origen (`original`), género y 37 etiquetas |
| `data/procesado/disaster_response_messages_es.csv` | Corpus preparado: 26 170 mensajes, con `message_es`, texto normalizado y una partición estratificada |
| `data/eda/` | Tablas del análisis exploratorio |

El campo `message` está en inglés. Solo 8 mensajes tienen un `original` realmente en español; el resto de `message_es` se obtuvo con [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es). La ficha del dataset declara la licencia como desconocida. El uso en este repositorio es académico.

## Cómo reproducirlo

```powershell
cd codigo
pip install -r requirements.txt
python 01_carga_inspeccion.py
python 02_eda.py
python 03_preprocesamiento.py
python 04_validacion_traduccion.py
```

`03_preprocesamiento.py` traduce unos 26 mil mensajes y en CPU tarda alrededor de 40 minutos. Si `data/procesado/disaster_response_messages_es.csv` ya está en el repositorio, no hace falta volver a ejecutarlo para revisar el corpus.

## Conclusiones

El corpus sirve para las cuatro preguntas, con tres límites que condicionan el entrenamiento:

- `request` es positivo en el 17,1 % de los mensajes y tiene una señal léxica clara (*ayuda*, *necesitamos*, *comida*, *agua*).
- `offer` solo tiene 117 positivos (0,45 %). En la partición de Hugging Face todos caen en test, así que no se puede entrenar con ese split. El archivo procesado trae una partición 80/10/10 estratificada (20 937 / 2 616 / 2 617) que reparte esas ofertas en 94, 12 y 11.
- La versión en español es casi toda traducción automática. En una muestra de 300 mensajes, la retrotraducción al inglés obtuvo chrF 71,0 y BLEU 47,4.

La propuesta, todavía sin entrenar, es una línea base TF-IDF con regresión logística y SVM lineal, y una comparación con BETO o DistilBERT multilingüe. La métrica principal será el F1 de la clase positiva.

## Licencia

El código de este repositorio se publica bajo la licencia MIT.

El dataset Disaster Response Messages no tiene una licencia identificada en su ficha de Hugging Face. No se relicencia aquí: se incluye solo para la evaluación del curso.
