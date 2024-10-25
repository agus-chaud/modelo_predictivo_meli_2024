import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform

# Función para crear nuevas características
def crear_nuevas_caracteristicas(df):
    # Relación entre precio y cantidad vendida
    df['precio_por_cantidad'] = df['price'] / df['sold_quantity']
    
    # Longitud del título del producto
    df['longitud_titulo'] = df['title'].str.len()
    
    # Cantidad de imágenes del producto
    df['cantidad_imagenes'] = df['pictures'].apply(len)
    
    # Días desde la fecha de creación del producto
    df['dias_desde_creacion'] = (pd.to_datetime('now') - pd.to_datetime(df['start_time'])).dt.days
    
    # Ratio de ventas completadas
    df['ratio_ventas_completadas'] = df['sold_quantity'] / df['available_quantity']
    
    # Categoría principal (primer nivel de la jerarquía de categorías)
    df['categoria_principal'] = df['category_id'].str.split('.').str[0]
    
    # Indicador de envío gratis
    df['envio_gratis'] = df['shipping_free_shipping'].astype(int)
    
    # Ratio de ventas diarias
    df['ratio_ventas_diarias'] = df['sold_quantity'] / df['dias_desde_creacion']
    
    # Indicador de producto oficial
    df['es_producto_oficial'] = df['official_store_id'].notna().astype(int)
    
    # Cantidad de atributos del producto
    df['cantidad_atributos'] = df['attributes'].apply(len)
    
    # Indicador de producto con garantía
    df['tiene_garantia'] = df['warranty'].notna().astype(int)
    
    # Descuento aplicado (si 'original_price' está disponible)
    if 'original_price' in df.columns:
        df['descuento'] = (1 - (df['price'] / df['original_price'])) * 100
        df['descuento'] = df['descuento'].clip(lower=0)  # Evitar descuentos negativos

    # Popularidad del vendedor (basada en ventas totales)
    df['popularidad_vendedor'] = df.groupby('seller_id')['sold_quantity'].transform('sum')

    # Precio promedio de los productos del vendedor
    df['precio_promedio_vendedor'] = df.groupby('seller_id')['price'].transform('mean')

    # Variedad de productos del vendedor
    df['variedad_productos_vendedor'] = df.groupby('seller_id')['id'].transform('count')

    # Indicador de producto con video
    df['tiene_video'] = df['video_id'].notna().astype(int)
    
    # exclusividad del producto
    df['ratio_precio_cantidad'] = df['price'] / (df['available_quantity'] + 1)
    # producto caro (¿premiun?) dentro de la categoria
    df['es_producto_caro'] = (df['price'] > df.groupby('categoria_principal')['price'].transform('mean')).astype(int)
    
    return df

# Cargar datos de entrenamiento
data = []
with open("C:/Agus/2024 2do cuatri/Análisis predictivo/competencia/train_data.jsonlines", 'r') as file:
    for line in file:
        data.append(json.loads(line))
df = pd.json_normalize(data, sep='_')

# Cargar datos de prueba
data_test = []
with open("C:/Agus/2024 2do cuatri/Análisis predictivo/competencia/test_data.jsonlines", 'r') as file:
    for line in file:
        data_test.append(json.loads(line))
df2 = pd.json_normalize(data_test, sep='_')

# Aplicar la función para crear nuevas características
df = crear_nuevas_caracteristicas(df)
df2 = crear_nuevas_caracteristicas(df2)

# Codificar variables categóricas
label_encoders_train = {}
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].astype(str)
    le = LabelEncoder()
    if column in df2.columns:
        valores_unicos = pd.concat([df[column], df2[column]]).apply(lambda x: str(x) if isinstance(x, list) else x).unique()
    else:
        valores_unicos = df[column].apply(lambda x: str(x) if isinstance(x, list) else x).unique()
    le.fit(valores_unicos)
    df[column] = le.transform(df[column])
    label_encoders_train[column] = le

# Aplicar la transformación a df2
for column in df2.select_dtypes(include=['object']).columns:
    df2[column] = df2[column].astype(str)
    if column in label_encoders_train:
        df2[column] = df2[column].map(lambda x: x if x in label_encoders_train[column].classes_ else 'desconocido')
        df2[column] = label_encoders_train[column].transform(df2[column])
    else:
        le = LabelEncoder()
        df2[column] = le.fit_transform(df2[column])
        label_encoders_train[column] = le

# Separar variables independientes (X) y dependientes (y) para entrenamiento
X = df.drop('condition', axis=1)
y = df['condition']

# Asegurarse de que x_test tenga las mismas columnas que X
x_test = df2[X.columns]

# Reemplazar infinitos con NaN
X = X.replace([np.inf, -np.inf], np.nan)
x_test = x_test.replace([np.inf, -np.inf], np.nan)

# Imputar valores faltantes
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_dist = {
    'iterations': randint(100, 500),
    'learning_rate': uniform(0.01, 0.2),
    'depth': randint(4, 10),
    'l2_leaf_reg': uniform(1, 10),
    'border_count': randint(32, 128),
    'bagging_temperature': uniform(0, 1),
    'random_strength': uniform(1e-9, 1),
    'scale_pos_weight': uniform(0.1, 10)
}

# Realizar la búsqueda aleatoria de hiperparámetros
random_search = RandomizedSearchCV(
    CatBoostClassifier(verbose=True, eval_metric='Accuracy'),
    param_distributions=param_dist,
    n_iter=40,
    cv=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Obtener el mejor modelo
mejor_modelo = random_search.best_estimator_

# Entrenar el mejor modelo con early stopping
mejor_modelo.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=30,
    verbose=100
)

# Preparar datos de prueba (df2) para predicción
X_test = df2.drop('condition', axis=1) if 'condition' in df2.columns else df2

# Reemplazar infinitos con NaN en X_test
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Transformar los datos de prueba usando el mismo imputador
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Hacer la predicción
y_pred = mejor_modelo.predict(X_test)

# Convertir las predicciones numéricas de vuelta a new o used
y_pred = label_encoders_train['condition'].inverse_transform(y_pred)
predictions_df = pd.DataFrame({'ID': range(1, len(y_pred) + 1), 'condition': y_pred})

# Guardar el DataFrame como un CSV
predictions_df.to_csv('C:/Agus/2024 2do cuatri/Análisis predictivo/competencia/predictions_catboost_optimizado.csv', index=False, sep=',')

print("Las predicciones se han guardado en 'predictions_catboost_optimizado.csv'")

# Imprimir los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

# Imprimir la precisión del modelo en el conjunto de validación
precision_validacion = mejor_modelo.score(X_val, y_val)
print(f"Precisión en el conjunto de validación: {precision_validacion:.4f}")