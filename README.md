Pipeline de machine learning para predecir si un producto de un marketplace es **nuevo** o **usado**, utilizando un modelo **CatBoost** con búsqueda de hiperparámetros.

## Flujo del script

### 1. Carga de datos
- Lee los archivos `train_data.jsonlines` y `test_data.jsonlines` en formato JSON Lines.
- Normaliza las estructuras JSON anidadas en DataFrames planos usando `pd.json_normalize`.

### 2. Feature engineering
Se generan nuevas características a partir de los datos originales:

| Feature | Descripción |
|---|---|
| `precio_por_cantidad` | Relación precio / cantidad vendida |
| `longitud_titulo` | Largo del título del producto |
| `cantidad_imagenes` | Número de imágenes del producto |
| `dias_desde_creacion` | Días transcurridos desde la publicación |
| `ratio_ventas_completadas` | Cantidad vendida / cantidad disponible |
| `categoria_principal` | Primer nivel de la jerarquía de categorías |
| `envio_gratis` | Indicador binario de envío gratis |
| `ratio_ventas_diarias` | Ventas promedio por día |
| `es_producto_oficial` | Si pertenece a una tienda oficial |
| `cantidad_atributos` | Número de atributos del producto |
| `tiene_garantia` | Si el producto tiene garantía |
| `descuento` | Porcentaje de descuento respecto al precio original |
| `popularidad_vendedor` | Ventas totales del vendedor |
| `precio_promedio_vendedor` | Precio promedio de los productos del vendedor |
| `variedad_productos_vendedor` | Cantidad de productos del vendedor |
| `tiene_video` | Si el producto tiene video |
| `ratio_precio_cantidad` | Precio / (cantidad disponible + 1) |
| `es_producto_caro` | Si el precio supera el promedio de su categoría |

### 3. Preprocesamiento
- Codificación de variables categóricas con `LabelEncoder` (ajustado sobre train + test para evitar categorías desconocidas).
- Reemplazo de valores infinitos por `NaN`.
- Imputación de valores faltantes con la mediana (`SimpleImputer`).

### 4. Entrenamiento del modelo
- División train/validación (80/20).
- Búsqueda aleatoria de hiperparámetros (`RandomizedSearchCV`) con 40 iteraciones y validación cruzada de 3 folds sobre un `CatBoostClassifier`.
- Re-entrenamiento del mejor modelo con early stopping (30 rondas sin mejora).

### 5. Predicción y exportación
- Genera predicciones sobre el conjunto de test.
- Convierte las etiquetas numéricas de vuelta a `new`/`used`.
- Guarda los resultados, con columnas `ID` y `condition`.
