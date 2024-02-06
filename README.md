> *Aplicación de técnicas de Machine Learning y Natural Language Processing para la extracción automática de metadatos bibliográficos desde documento académicos de la Universidad Mayor de San Andrés*

Este repositorio contiene el codigo desarrollado como parte del proyecto de grado "Aplicación de técnicas de Machine Learning y Natural Language Processing para la extracción automática de metadatos bibliográficos en documento académicos de la Universidad Mayor de San Andrés" para el pregrado en Ingeniería Electrónica de la Universidad Mayor de San Andrés.

El proyecto se encuentra dividido en dos partes:

1. **Extractor de metadatos**: Consiste en un pipeline de Reconocimiento de Entidades Nombradas (NER), el cual fue entrenado para extraer metadatos desde la caratula de documentos académicos de la UMSA.
2. **Generador de palabras clave**: Consiste en un modelo LLM el cual ha sido Ajustado para generar palabras clave a partir del texto del resumen de los documentos académicos de la UMSA.

## Estructura del repositorio

El repositorio se encuentra estructurado de la siguiente manera:

- `extractor_de_metadatos/`: Contiene el código de entrenamiento del extractor de metadatos.
- `generator_de_palabras_clave/`: Contiene el código de entrenamiento del generador de palabras clave.
- `demo/`: Contiene el código de la aplicación web que permite probar los modelos entrenados.

## Modelos entrenados y datasets

Los modelos entrenados se encuentran disponibles en 🤗 Hugging Face Hub y pueden ser descargados desde los siguientes enlaces:

- [Modelo extractor de metadatos](https://huggingface.co/riturralde/es_metaextract_umsa_v2)
- [Modelo generador de palabras clave](https://huggingface.co/riturralde/es_metaextract_umsa_adapter_v1)