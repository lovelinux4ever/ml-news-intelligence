# 🧠 ML News Intelligence Engine

Este repositorio alberga el núcleo de un motor de procesamiento de lenguaje natural (NLP) especializado en la detección y validación semántica de noticias tecnológicas.

## ⚖️ Nota sobre el Dataset y Cumplimiento
Por motivos de **propiedad intelectual y cumplimiento de términos de servicio (TOS)** de las fuentes de origen, el dataset de entrenamiento original no se incluye en este repositorio público. 

Este proyecto se centra en la exposición de los **artefactos del modelo** y la demostración de la arquitectura de ingeniería de datos subyacente.

## ⚙️ Especificaciones Técnicas
- **Modelo**: Vectorizador TF-IDF con vocabulario optimizado de 3,000 dimensiones.
- **Lógica de Match**: Cálculo de Similitud de Coseno para la identificación de entidades en estructuras DOM.
- **Rendimiento**: Matriz de características comprimida para alta eficiencia en memoria.

## 🚀 Demo de Validación
El script `demo_engine.py` permite verificar la capacidad del modelo para reconocer titulares específicos dentro de bloques de texto (simulación de scraping), demostrando una precisión de match del **77%**.
