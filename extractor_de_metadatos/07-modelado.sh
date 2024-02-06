#!/bin/bash

entities="all advisors authors department faculty title"

for entity in $entities:
do
    echo "Entrenando modelo NER para la entidad ${entity}..."

    # Eliminar archivos de log de entrenamiento anteriores
    rm "${entity}_training_log.jsonl" || true

    # Entrenar modelo NER
    python -m spacy train config.cfg --custom.entity "${entity}" --output "./${entity}_ner_model"
done
