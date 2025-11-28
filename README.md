# Fraud Detection in Banking Transactions using Machine Learning

Este repositorio contiene el desarrollo de un modelo de machine learning para la detección de transacciones bancarias fraudulentas. 
El proyecto aborda la problemática del fraude digital en entidades financieras, proponiendo un enfoque predictivo que supera las 
limitaciones de los sistemas tradicionales basados en reglas.

## Contenido
- **Documentación académica**: Presentación y reporte del proyecto con contexto, objetivos, metodología, resultados y conclusiones.
- **Código fuente (Python/Scikit-learn)**: Implementación del modelo con etapas de preprocesamiento, análisis de correlaciones, 
  escalamiento de variables, codificación de atributos categóricos y entrenamiento de algoritmos.
- **Modelos evaluados**: Regresión Logística y Random Forest, con optimización de hiperparámetros.
- **Resultados**: El modelo Random Forest alcanzó un recall de 95%, detectando casi todos los casos de fraude en el dataset.
- **Implicancias éticas**: Consideraciones sobre privacidad de datos, sesgos, transparencia y seguridad frente a ataques adversarios.

## Objetivo
Desarrollar un sistema predictivo capaz de identificar patrones en transacciones bancarias y anticipar posibles fraudes, 
reduciendo falsos negativos y fortaleciendo la confianza en los sistemas financieros.

## Dataset
Se utilizó un conjunto de datos público (`fraud_data.csv`) con información de transacciones bancarias, incluyendo atributos 
temporales, geográficos, demográficos y financieros.

## Tecnologías utilizadas
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
- Algoritmos de clasificación supervisada
- GridSearchCV para optimización de hiperparámetros
