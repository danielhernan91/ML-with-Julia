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
