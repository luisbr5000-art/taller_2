# Contexto del proyecto - Gestión de Datos (Taller 2: Limpieza de Datos)
- Objetivo: diagnóstico y limpieza del dataset Titanic (faltantes, atípicos, duplicados y redundancia) en un notebook Jupyter.
- Instrucciones del taller: instrucciones/Taller_2_Limpieza_de_Datos.pdf.
- Entorno: contenedor con Jupyter Lab (imagen quay.io/jupyter/scipy-notebook), ejecutado con Docker (Dockerfile, docker-compose.yml, requirements.txt).
- Carpeta de trabajo: ./work (en el contenedor: /home/jovyan/work).
- Notebook de trabajo: work/taller2_<apellidos>_<nombres>.ipynb (nombre pendiente). Se construye por secciones:
agrega cada sección al final y no modifiques las anteriores.
- Datos: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
- Diccionario de columnas: https://www.kaggle.com/c/titanic/data
- Librerías permitidas: pandas, numpy, scipy, matplotlib, seaborn, ydata-profiling.
- Entregables: notebook ejecutado de principio a fin, work/titanic_limpio.csv, reporte HTML de ydata-profiling y archivos del entorno.

# Reglas para el agente
- Responde y comenta el código en español.
- Una tarea por vez, con cambios pequeños y explicados.
- Nunca inventes resultados: todo número debe estar respaldado y salir de ejecutar código.
- No instales librerías nuevas sin preguntar.
- No envíes datos fuera de la máquina local.
- Las interpretaciones y conclusiones las escribe el usuario; el agente solo puede revisarlas.
- Registrar cada prompt en la bitácora agéntica (Rol, Tarea, Formato; modelo; créditos; qué se corrigió).
