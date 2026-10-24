# Taller 2: Limpieza de Datos

**Gestión de Datos · Módulo 1 · 2026**

**Integrantes:** Luis Alberto Banda, Maria Camila Rincón, Juan Pablo Marin, Gabriel Camilo Velasquez y Juan Camilo Guerrero

## Objetivo

Diagnosticar la calidad del dataset **Titanic** y limpiarlo para que pueda usarse luego en un **modelo predictivo** de supervivencia (`Survived`). El trabajo cubre los cuatro problemas típicos de calidad: valores faltantes, valores atípicos, duplicados y redundancia.

El taller tiene un hilo conductor: el **Ejercicio 1** diagnostica cada columna y propone una acción de mitigación, y el **Ejercicio 2** ejecuta esas acciones, citando en cada paso el hallazgo que atiende.

- **Datos:** https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv (891 filas × 12 columnas)
- **Diccionario de columnas:** https://www.kaggle.com/c/titanic/data

Todas las cifras de este documento salen de ejecutar el notebook y se reportan con todos sus decimales, sin redondear. Las cifras del diagnóstico que en sus celdas originales aparecen redondeadas se recalculan con todos sus decimales en la sección **4.3** del notebook.

## Estructura del repositorio

| Archivo | Contenido |
|---|---|
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Entorno reproducible: Jupyter Lab sobre la imagen `quay.io/jupyter/scipy-notebook` con pandas, numpy, scipy, matplotlib, seaborn y ydata-profiling |
| `work/Taller_2_Gestion_de_Datos.ipynb` | Notebook con los dos ejercicios, ejecutado de principio a fin |
| `work/titanic.csv` | Copia local del dataset original |
| `work/titanic_profiling.html` | Reporte exploratorio de ydata-profiling (dataset original) |
| `work/titanic_limpio.csv` | Dataset limpio (entregable del Ejercicio 2) |
| `work/figuras/` | Figuras de antes y después de `Age` y `Fare` |
| `BITACORA.md` | Bitácora agéntica: prompts usados, modelo, créditos y correcciones |

## Cómo ejecutarlo

```bash
docker compose up --build
```

Luego abrir http://127.0.0.1:8888 (solo accesible desde la máquina local) y ejecutar `work/Taller_2_Gestion_de_Datos.ipynb` completo (*Run → Run All Cells*). La carpeta `work` está montada en el contenedor como `/home/jovyan/work`, así que todos los archivos generados quedan en el repositorio.

### Organización del notebook

| Sección | Contenido |
|---|---|
| 4.1 | Carga, primeras filas, shape, dtypes y reporte de ydata-profiling |
| 4.2 | Tabla de diagnóstico por columna con la acción de mitigación |
| 4.3 | Complemento del diagnóstico: cifras con todos sus decimales y verificación de la tarifa por tiquete |
| 4.4 | Columna más problemática (entregable del Ejercicio 1) |
| 5.1 – 5.6 | Limpieza: duplicados, inconsistencias, faltantes, atípicos, redundancia y exportación |
| 5.7 – 5.8 | Resumen de la limpieza y análisis (entregable del Ejercicio 2) |

---

## Ejercicio 1: Diagnóstico de calidad de los datos

### Carga y vistazo inicial

Se cargó el dataset con `pd.read_csv` (891 filas × 12 columnas), se revisaron las 10 primeras filas y los tipos de dato, y se generó el reporte de ydata-profiling (`work/titanic_profiling.html`). El reporte se usó solo como apoyo: cada hallazgo se verificó con una celda propia.

### Hallazgos por verificación

**Valores únicos o constantes.** No hay columnas constantes. `PassengerId` y `Name` tienen un valor distinto en cada una de las 891 filas: son identificadores sin valor predictivo. `Ticket` tiene 681 valores distintos, así que funciona casi como un identificador.

**Valores faltantes.**

| Columna | Nulos | % nulos | Mecanismo probable | Evidencia |
|---|---|---|---|---|
| `Cabin` | 687 | 77.10437710437711 | MAR (posible MNAR) | Falta en el 97.5560081466395 % de 3.ª clase frente al 18.51851851851852 % de 1.ª, y en el 87.61384335154827 % de quienes no sobrevivieron frente al 60.23391812865497 % de quienes sí |
| `Age` | 177 | 19.865319865319865 | MAR | Falta en el 27.698574338085542 % de 3.ª clase frente al 5.978260869565218 % de 2.ª, y en el 22.768670309653917 % de quienes no sobrevivieron frente al 15.204678362573098 % de quienes sí |
| `Embarked` | 2 | 0.2244668911335578 | MCAR | Solo 2 filas, sin patrón (ambas de 1.ª clase, mismo tiquete 113572 y cabina B28) |

**Faltantes disfrazados.** `Fare` tiene 15 valores en 0 (1.6835016835016836 % de las filas), una tarifa no plausible que se trata como faltante. `Ticket` tiene 4 tiquetes sin ningún dígito (`"LINE"`). No se encontraron marcadores de texto como `"?"` o `"Unknown"`.

**Valores atípicos por atributo (Tukey: Q1 − 1,5·IQR, Q3 + 1,5·IQR).**

| Columna | Límite inferior | Límite superior | Atípicos | % atípicos |
|---|---|---|---|---|
| `Age` | -6.6875 | 64.8125 | 11 | 1.2345679012345678 |
| `Fare` | -26.724 | 65.6344 | 116 | 13.019079685746352 |
| `SibSp` | -1.5 | 2.5 | 46 | 5.162738496071829 |
| `Parch` | 0.0 | 0.0 | 213 | 23.905723905723907 |

En `Parch` el IQR es 0, así que Tukey marca como atípico a todo pasajero que viaja con padres o hijos: no son errores sino un efecto de la regla. En `Age`, `SibSp` y `Fare` los atípicos son valores reales (personas mayores, familias numerosas y tarifas de primera clase).

**Registros atípicos.**
- 82 filas tienen 2 o más atributos atípicos al mismo tiempo. Están repartidas por igual entre 1.ª y 3.ª clase (41 y 41); en 44 de ellas el pasajero viaja con 3 o más hermanos o cónyuges, y 37 son menores de 18 años.
- Al calcular Tukey dentro de cada clase aparecen 52 tarifas atípicas en 3.ª clase. La verificación de 4.3.3 explica por qué: **`Fare` es el precio del tiquete completo, no por persona**. De 134 tiquetes compartidos, solo 1 tiene más de una tarifa, y las 52 tarifas atípicas de 3.ª clase son todas de tiquetes compartidos (por ejemplo, los 7 pasajeros del tiquete `CA. 2343` pagan 69.55 cada uno).

**Redundancia.** `SibSp` y `Parch` se solapan con el tamaño de familia (`SibSp + Parch + 1`): correlaciones de Spearman de 0.8491645107690946 y 0.8012240815976783. `Pclass` y `Fare` tienen una correlación de Spearman de -0.6880316726256098, pero miden cosas distintas (clase del tiquete y precio pagado).

### Columna más problemática: `Cabin`

`Cabin` es la columna más problemática: le falta el 77.10437710437711 % de los valores, y esos faltantes no son al azar, porque dependen de la clase y de la supervivencia. Ninguna corrección es buena:
- **Imputarla** obligaría a inventar la mayoría de una columna con 147 valores distintos.
- **Eliminar las filas con nulo** dejaría 204 pasajeros, el 86.27450980392157 % de ellos de primera clase.
- **Eliminar la columna** pierde la información. Lo único recuperable es si el pasajero tenía o no cabina registrada.

---

## Ejercicio 2: Limpieza del dataset

Se trabajó sobre una copia (`df_limpio`) para poder comparar siempre contra el original. La limpieza sigue la tabla de mitigación del Ejercicio 1:

| Hallazgo del diagnóstico | Acción en la limpieza | Sección |
|---|---|---|
| `PassengerId` tiene un valor por fila: un pasajero repetido no se vería como duplicado | Buscar duplicados ignorando `PassengerId` | 5.1 |
| Faltantes disfrazados: 15 `Fare = 0` y 4 tiquetes `"LINE"` | Convertirlos a `NaN` con Boolean Indexing | 5.2 |
| `Cabin`: columna más problemática | Crear `TieneCabina` y eliminar la columna con `dropna` | 5.3 |
| `Age`: MAR | Imputar con la mediana por `Pclass` × `Sex` y crear `EdadFaltante` | 5.3 |
| `Embarked`: MCAR | Imputar con la moda | 5.3 |
| `Fare`: ceros convertidos a `NaN`, tarifa ligada a la clase | Imputar con la mediana por `Pclass` | 5.3 |
| `Fare`: 116 atípicos reales (propuesta: conservar o transformar) | Conservar las filas y acotar con winsorización p5–p95 | 5.4 |
| Atípicos de `Age`, `SibSp`, `Parch` plausibles | Conservarlos | 5.4 |
| Registros atípicos; `Fare` es precio por tiquete | Conservar las filas; evaluar la tarifa por persona; guardar `pasajeros_tiquete` | 5.4 |
| `SibSp` y `Parch` se solapan con el tamaño de familia | Crear `tamano_familia` y eliminar `SibSp` y `Parch` | 5.5 |
| `Name` identificador; `Ticket` casi identificador | Extraer `Titulo` y eliminar `Name` y `Ticket` | 5.5 |
| `PassengerId` identificador; `Pclass`–`Fare` correlacionadas pero distintas | Conservar `PassengerId` como clave; conservar `Pclass` y `Fare` | 5.5 |
| `Sex`: codificar como categórica | Convertir `Sex`, `Embarked` y `Titulo` a tipo `category` | 5.5 |

Las cinco funciones que pide el enunciado se usan así: `drop_duplicates` en 5.1, Boolean Indexing en 5.2, `dropna` y `fillna` en 5.3, y `drop` en 5.5.

### 5.1 y 5.2: Duplicados e inconsistencias

- **Duplicados:** no hay duplicados, ni exactos ni ignorando `PassengerId`; el shape se mantiene en (891, 12).
- **Faltantes disfrazados:** con máscaras booleanas, los 15 `Fare = 0` y los 4 `"LINE"` pasaron a `NaN`.
- **Reglas de dominio:** se verificaron 7 reglas (valores válidos de sexo, clase, supervivencia y puerto; edades entre 0 y 100; `SibSp` y `Parch` no negativos; tarifa no negativa). Ninguna fila las incumple.

### 5.3 Faltantes (punto 8)

| Columna | Mecanismo | Decisión | Por qué |
|---|---|---|---|
| `Cabin` | MAR (posible MNAR) | Eliminar la columna (umbral de 50 % de no nulos) y crear `TieneCabina` | Es la columna más problemática: solo 204 de 891 valores observados. No se extrae la cubierta porque faltaría en las mismas filas. |
| `Age` | MAR | Imputar con la mediana por `Pclass` × `Sex` y crear `EdadFaltante` | Eliminar 177 filas sesgaría la muestra hacia las clases altas; bajo MAR es válido imputar condicionando en variables observadas |
| `Embarked` | MCAR | Imputar con la moda (`S`) | Solo 2 filas sin patrón; comparten tiquete, así que quedan con el mismo puerto |
| `Fare` (ceros) | MAR | Imputar con la mediana de su clase (61.9792, 15.0229 y 8.05 para 1.ª, 2.ª y 3.ª) | Los ceros no se reparten igual entre clases (2.314814814814815 %, 3.260869565217391 % y 0.8146639511201629 % de cada clase) y la tarifa depende de la clase |

**Efecto en `Age`:**

| | Antes de imputar | Después de imputar | Diferencia |
|---|---|---|---|
| Media | 29.69911764705882 | 29.11242424242424 | -0.5866934046345804 |
| Mediana | 28.0 | 26.0 | -2.0 |
| Desviación estándar | 14.526497332334044 | 13.304424074688153 | -1.2220732576458904 |

La media cambia poco, pero la dispersión baja: los 177 pasajeros imputados quedan concentrados en solo seis valores (las medianas de cada grupo), lo que crea un pico artificial entre 25 y 30 años ([figura](work/figuras/age_antes_despues.png)). La imputación por grupo respeta el mecanismo MAR, pero no recupera la variabilidad real; por eso se conserva el indicador `EdadFaltante`.

### 5.4 Atípicos (punto 9)

El diagnóstico propuso **conservar o transformar** los atípicos de `Fare`, porque son tarifas reales. De las dos opciones del enunciado se eligió la **winsorización** (percentiles 5 = 7.2292 y 95 = 112.07915), que hace justamente eso:
- **No elimina filas.**
- **Conserva el orden de las tarifas:** la correlación de Spearman entre la serie antes y después es 0.9998221135170301.
- **Solo acota las colas:** modifica 85 valores (40 abajo y 45 arriba). Tukey, en cambio, habría recortado 116, porque su límite superior es 66.3.

| | Original | Tras imputar ceros | Winsorizada p5–p95 |
|---|---|---|---|
| Media | 32.204207968574636 | 32.68931840628507 | 28.22123552188552 |
| Mediana | 14.4542 | 14.5 | 14.5 |
| Desviación estándar | 49.693428597180905 | 49.61163877217443 | 29.14891277694091 |
| Mínimo | 0.0 | 4.0125 | 7.2292 |
| Máximo | 512.3292 | 512.3292 | 112.07915 |

La media baja 3.9829724466891143 y la desviación estándar 20.544515820239994, mientras la mediana no cambia por el recorte ([figura](work/figuras/fare_antes_despues.png)).

**Registros atípicos y tarifa de grupo.**
- **Registros atípicos:** las 82 filas con varios atributos atípicos se conservan todas, y la winsorización acotó la tarifa de 25 de ellas.
- **Tarifas atípicas de 3.ª clase:** la winsorización no acotó ninguna de las 52, porque son altas solo frente a su clase, al ser precios de tiquetes de grupo.
- **Tarifa por persona (`Fare` / pasajeros del tiquete):** se evaluó como corrección y se descartó. Reduce los atípicos dentro de la clase en 1.ª (de 20 a 9), pero los aumenta en 2.ª (de 7 a 28) y en 3.ª (de 52 a 103). En su lugar, se conserva `Fare` como precio del tiquete y se agrega `pasajeros_tiquete`, para que el modelo pueda interpretar la tarifa según el tamaño del grupo.

Los atípicos de `Age` (11), `SibSp` (46) y `Parch` (213) se conservan sin cambios, como propuso el diagnóstico.

### 5.5 Redundancia y data leakage (punto 10)

| Columna | Decisión | Por qué |
|---|---|---|
| `SibSp`, `Parch` | Eliminar | `tamano_familia` es su suma exacta (\|rho\| de Spearman de 0.8491645107690946 y 0.8012240815976783 con ella): conservar las tres crea colinealidad perfecta |
| `tamano_familia` | Conservar | Resume `SibSp` y `Parch` |
| `pasajeros_tiquete` | Conservar | Correlacionada con `tamano_familia` (\|rho\| = 0.6736039640418285) pero no la duplica: un tiquete puede agrupar a personas que no son familia. Se relaciona más con `Survived` (0.18652946148913782) que `tamano_familia` (0.16546291168480365) |
| `Name` | Eliminar tras extraer `Titulo` | Identificador; el título resume sexo, edad y estado civil |
| `Ticket` | Eliminar | Casi identificador; su información útil quedó en `pasajeros_tiquete` |
| `PassengerId` | Conservar como clave | No se usa en el modelo, pero permite rastrear cada fila |
| `Pclass`, `Fare` | Conservar ambas | \|rho\| = 0.7296406038722933 en los datos limpios, pero miden cosas distintas; un modelo lineal sufriría colinealidad entre ellas. `TieneCabina` también está ligada a la clase (\|rho\| = 0.6832907054005039 con `Pclass`) |
| `Sex`, `Embarked`, `Titulo` | Codificar como `category` | Propuesta del diagnóstico para `Sex`, extendida a las demás variables de texto |

**Columnas con riesgo de data leakage:**

| Columna | % que sobrevivió | Riesgo |
|---|---|---|
| `TieneCabina` | 66.66666666666666 % con cabina frente a 29.985443959243085 % sin cabina | Si el registro de cabina se completó después del naufragio, el indicador codifica parte del resultado |
| `EdadFaltante` | 29.37853107344633 % con edad faltante frente a 40.61624649859944 % con edad registrada | Mismo argumento: el faltante es más frecuente entre quienes no sobrevivieron |
| `Age`, `Fare`, `pasajeros_tiquete` | — | Las medianas, los percentiles y los conteos por tiquete se calcularon con las 891 filas; al separar entrenamiento y prueba deben recalcularse solo con entrenamiento |

### 5.6 Resultado final

- **Filas eliminadas:** 0 (no hubo duplicados ni registros inválidos, y los registros atípicos se conservaron a propósito).
- **Columnas eliminadas (5):** `Cabin`, `Name`, `Ticket`, `SibSp`, `Parch`.
- **Columnas agregadas (5):** `TieneCabina`, `EdadFaltante`, `pasajeros_tiquete`, `tamano_familia`, `Titulo`.
- **Valores imputados (194):** 177 en `Age`, 15 en `Fare` y 2 en `Embarked`.
- **Valores acotados (85):** las colas de `Fare`.
- **Dataset limpio:** 891 filas × 12 columnas, sin nulos (`work/titanic_limpio.csv`).

| Columna | dtype |
|---|---|
| `PassengerId` | int64 |
| `Survived` | int64 |
| `Pclass` | int64 |
| `Sex` | category |
| `Age` | float64 |
| `Fare` | float64 |
| `Embarked` | category |
| `TieneCabina` | int64 |
| `EdadFaltante` | int64 |
| `pasajeros_tiquete` | int64 |
| `tamano_familia` | int64 |
| `Titulo` | category |

El CSV no guarda el tipo `category`: al leer `titanic_limpio.csv` hay que indicarlo con `dtype={"Sex": "category", "Embarked": "category", "Titulo": "category"}`.

## Conclusiones del analista

Cada conclusión indica la celda del notebook que la respalda, según el número que Jupyter muestra a la izquierda de cada celda de código.

1. **No fue necesario eliminar ningún pasajero.** No se encontraron registros duplicados, ni siquiera ignorando el identificador del pasajero (celda 20), y ningún registro tenía valores imposibles, como edades negativas o clases inexistentes (celda 23). El dataset limpio conserva los 891 pasajeros del original (celda 46).

2. **La columna de cabina se eliminó por exceso de nulos, pero se conservó su información más útil.** Le faltaba el 77.10437710437711 % de los valores (celda 13): solo 204 pasajeros tenían cabina registrada, cuando el criterio fijado exigía al menos 446 (celda 25). Eliminar a los pasajeros sin cabina tampoco era opción, porque el 86.27450980392157 % de los que quedarían serían de primera clase (celda 18). Antes de eliminar la columna se creó un indicador de si el pasajero tenía o no cabina registrada (celda 25).

3. **La edad se completó según la clase y el sexo del pasajero, porque sus faltantes no son aleatorios, aunque eso redujo su variabilidad.** La edad falta mucho más en tercera clase (27.698574338085542 %) que en segunda (5.978260869565218 %) (celda 13). Por eso las 177 edades faltantes se completaron con la mediana de los pasajeros de la misma clase y sexo (celda 28). La edad promedio cambió poco, de 29.69911764705882 a 29.11242424242424 años, pero la desviación estándar bajó de 14.526497332334044 a 13.304424074688153 (celda 29), y aparece una concentración artificial de pasajeros entre 25 y 30 años (celda 30). Se agregó un indicador que marca a qué pasajeros se les completó la edad.

4. **Los faltantes disfrazados se identificaron y corrigieron antes de completar los datos.** Las 15 tarifas registradas en cero y los 4 tiquetes con el texto "LINE" se trataron como datos faltantes (celdas 21 y 22). Las tarifas en cero se completaron con la tarifa mediana de su clase: 61.9792 en primera, 15.0229 en segunda y 8.05 en tercera (celda 27). Los 2 pasajeros sin puerto de embarque, cuyo faltante es aleatorio, se asignaron al puerto más frecuente, Southampton (celda 26).

5. **El recorte de las tarifas extremas redujo su peso sin perder información.** Se acotaron las tarifas entre los percentiles 5 y 95, lo que modificó 85 valores; con el criterio de Tukey se habrían recortado 116. El orden de las tarifas se mantuvo prácticamente igual: la correlación de Spearman entre antes y después es 0.9998221135170301 (celda 32). La tarifa promedio bajó de 32.204207968574636 a 28.22123552188552, y la desviación estándar de 49.693428597180905 a 29.14891277694091 (celda 33). La comparación gráfica está en la celda 34.

6. **Las tarifas atípicas dentro de cada clase no son errores, sino precios de tiquetes de grupo.** Los pasajeros que comparten tiquete pagan la misma tarifa (de 134 tiquetes compartidos, solo 1 tiene dos tarifas distintas), y las 52 tarifas atípicas de tercera clase corresponden todas a tiquetes compartidos (celda 16). Se evaluó dividir la tarifa entre los pasajeros de cada tiquete, pero se descartó porque empeoraba el problema: los casos atípicos subían de 7 a 28 en segunda clase y de 52 a 103 en tercera (celda 35). En su lugar se guardó cuántos pasajeros comparten cada tiquete. Los 82 pasajeros con varios valores atípicos se conservaron, y el recorte no modificó ninguna de las 52 tarifas atípicas de tercera clase (celda 36).

7. **Se eliminó la información repetida sin perder información útil.** Se creó el tamaño de la familia a bordo como la suma de hermanos o cónyuges, padres o hijos, y el propio pasajero (celda 38). Esa nueva variable está muy relacionada con las dos originales, con correlaciones de Spearman de 0.8491645107690946 y 0.8012240815976783 (celda 40). Por eso se eliminaron las dos originales, junto con el nombre y el número de tiquete, que funcionan como identificadores (celda 43). Del nombre se extrajo antes el título (Mr, Mrs, Miss, etc.) (celda 42). El resultado final tiene 891 pasajeros, 12 columnas y ningún dato faltante (celda 45).
