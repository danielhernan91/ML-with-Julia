########################### ENFOQUE INFERENCIAL EN JULIA ###########################
# 1. INSTALAR E IMPORTAR PAQUETES
# Añadimos MLDataUtils para dividir los datos.
# using Pkg; Pkg.add("GLM"); Pkg.add("DataFrames"); Pkg.add("MLDataUtils"); Pkg.add("Statistics");
# --------------------------------------------------------------------
using GLM, DataFrames, MLDataUtils, Statistics

# 2. GENERAR DATOS DE EJEMPLO
# --------------------------------------------------------------------
num_puntos = 100
X = 1:num_puntos
y = 2.5 .* X .+ 10 .+ randn(num_puntos) .* 15 # y = 2.5x + 10 + ruido
df = DataFrame(X=X, y=y)

# 3. DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA (TRAIN/TEST SPLIT)
# Usaremos un 80% para entrenamiento y un 20% para prueba.
# shuffleobs mezcla los datos antes de dividirlos para evitar sesgos.
# --------------------------------------------------------------------
(train_df, test_df) = MLDataUtils.splitobs(shuffleobs(df), at = 0.8)

println("Número de datos de entrenamiento: ", nrow(train_df))
println("Número de datos de prueba: ", nrow(test_df))
println("-"^40)

# 4. ENTRENAR EL MODELO (SOLO CON DATOS DE ENTRENAMIENTO)
# El modelo solo "aprende" de los datos de train_df.
# --------------------------------------------------------------------
modelo = lm(@formula(y ~ X), train_df)

println("Resultados del Modelo Entrenado:")
println(modelo)
println("-"^40)

# 5. HACER PREDICCIONES SOBRE EL CONJUNTO DE PRUEBA
# Ahora usamos el conjunto de prueba, que el modelo no ha visto.
# --------------------------------------------------------------------
predicciones_test = GLM.predict(modelo, test_df)

# 6. EVALUAR EL RENDIMIENTO DEL MODELO
# Comparamos las predicciones con los valores reales del conjunto de prueba.
# Una métrica común es el Error Cuadrático Medio (MSE).
# --------------------------------------------------------------------
y_real_test = test_df.y
mse = mean((predicciones_test .- y_real_test).^2)

println("Evaluación del modelo con datos de prueba:")
println("Error Cuadrático Medio (MSE): ", round(mse, digits=2))
println("Esto nos da una idea de qué tan lejos, en promedio, están nuestras predicciones del valor real.")

########################### ENFOQUE OPTIMIZACION EN JULIA ###########################

using MLJ, DataFrames, Statistics

# MLJ requiere que los datos y el objetivo estén separados
# Usaremos los mismos datos de antes
num_puntos = 100
X_raw = 1:num_puntos
y = 2.5 .* X_raw .+ 10 .+ randn(num_puntos) .* 15
X = DataFrame(X=X_raw)

# 1. CARGAR EL MODELO Y DEFINIR EL RANGO DE BÚSQUEDA
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
model_instance = RidgeRegressor() # <-- Create an instance here

# Use the instance to define the range for the :lambda hyperparameter
rango_lambda = range(model_instance, :lambda, lower=0.01, upper=100.0, scale=:log10)

# 2. CONFIGURAR LA ESTRATEGIA DE OPTIMIZACIÓN
estrategia_cv = CV(nfolds=6)

modelo_RegML = TunedModel(
    model=model_instance, # <-- Reuse the instance here
    resampling=estrategia_cv,
    tuning=Grid(resolution=10),
    range=rango_lambda,
    measure=rms
)
# Creamos el modelo "RegML"
modelo_RegML = TunedModel(
    model=RidgeRegressor(),
    resampling=estrategia_cv,
    tuning=Grid(resolution=10), # Probará 10 valores de lambda en el rango
    range=rango_lambda,
    measure=rms # Usaremos la Raíz del Error Cuadrático Medio como métrica
)

# 3. ENTRENAR Y OPTIMIZAR
# Al hacer "fit", MLJ entrena el modelo con cada valor de lambda y encuentra el mejor
maquina_regresion = machine(modelo_RegML, X, y)
fit!(maquina_regresion)

# 4. VER LOS RESULTADOS
reporte = MLJ.report(maquina_regresion)
mejor_lambda = reporte.best_model.lambda

println("El mejor hiperparámetro (lambda) encontrado es: ", round(mejor_lambda, digits=4))

# Con este lambda optimizado, ahora tienes el mejor modelo Ridge posible

# 5. HACER PREDICCIONES CON EL MODELO OPTIMIZADO
# Una vez entrenada la máquina, puedes usar `predict` directamente sobre ella.
# MLJ usará automáticamente el mejor modelo encontrado durante el tuning.
# --------------------------------------------------------------------
predicciones = MLJ.predict(maquina_regresion, X)
println("Predicciones para las primeras 5 filas:")

println(round.(predicciones[1:5], digits=2))
