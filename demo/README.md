# MetaExtract-UMSA Demo

MetaExtract-UMSA es un conjunto de modelos de ML para la extracción de metadatos de documentos académicos publicados por estudiantes de la Universidad Mayor de San Andrés (UMSA).
 
Este repositorio contiene una aplicación de demostración que permite probar los modelos de extracción de metadatos de MetaExtract-UMSA. La misma esta desarrollada en Python y utiliza el framework [Gradio](https://www.gradio.app/) para la creación de la interfaz gráfica.

## Instalación

Para ejecutar la demostración, se debe instalar las dependencias de Python:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install auto-gptq==0.4.2 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

## Ejecución

Para ejecutar la demostración, se debe ejecutar el siguiente comando:

```bash
python demo.py
```

En caso de no contar con una GPU, se puede ejecutar la aplicación sin el modelo de generación de palabras clave, utilizando el siguiente comando:

```bash
python demo.py --no-keywords
```


