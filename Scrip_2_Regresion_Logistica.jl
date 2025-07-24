############## ENFOQUE INFERENCIAL EN JULIA PARA REGRESIÓN LOGÍSTICA ###########################

using GLM, DataFrames, MLDataUtils, Statistics

# 1. GENERAR DATOS DE CLASIFICACIÓN
# Crearemos datos donde la probabilidad de ser '1' depende de 'X'.
num_puntos = 200
X1 = randn(num_puntos)
X2 = randn(num_puntos)
# La probabilidad 'p' de que y=1 aumenta con X1 y disminuye con X2
p = 1.0 ./ (1.0 .+ exp.(-(2.0 .* X1 .- 3.0 .* X2 .+ 1.0)))
y = rand(num_puntos) .< p # y será true/false (1/0)

df = DataFrame(X1=X1, X2=X2, y=y)

# 2. DIVIDIR DATOS (ENTRENAMIENTO Y PRUEBA)
(train_df, test_df) = MLDataUtils.splitobs(shuffleobs(df), at=0.8)

# 3. ENTRENAR EL MODELO LOGÍSTICO
# Usamos glm() con la familia Binomial() y el enlace LogitLink()
modelo_logit = glm(@formula(y ~ X1 + X2), train_df, Binomial(), LogitLink())

# 4. HACER PREDICCIONES
# predict() devuelve probabilidades (un número de 0.0 a 1.0)
probabilidades = GLM.predict(modelo_logit, test_df)

# Convertimos probabilidades a clases (0 o 1) usando un umbral de 0.5
predicciones_clase = ifelse.(probabilidades .> 0.5, true, false)

# 5. EVALUAR EL MODELO
# Para clasificación, usamos métricas como la "accuracy" (precisión)
y_real = test_df.y
accuracy = mean(predicciones_clase .== y_real)

println("Resultados del Modelo Logístico:")
println(modelo_logit)
println("\nPrecisión (Accuracy) en datos de prueba: ", round(accuracy * 100, digits=2), "%")

using MLJ, DataFrames, Statistics

############################# ENFOQUE OPTIMIZACIÓN EN JULIA PARA REGRESIÓN LOGÍSTICA ###########################

# 1. PREPARAR DATOS
# Los datos son los mismos, pero MLJ prefiere el objetivo 'y' como tipo Categorial.
num_puntos = 200
X1 = randn(num_puntos)
X2 = randn(num_puntos)
p = 1.0 ./ (1.0 .+ exp.(-(2.0 .* X1 .- 3.0 .* X2 .+ 1.0)))
y_bool = rand(num_puntos) .< p
X = DataFrame(X1=X1, X2=X2)
y = categorical(y_bool) # Convertir a Categorial es buena práctica en MLJ

# 2. DIVIDIR DATOS
(train_rows, test_rows) = partition(eachindex(y), 0.8, shuffle=true)

# 3. CARGAR MODELO Y CREAR LA MÁQUINA
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model = LogisticClassifier()
mach = machine(model, X, y)

# 4. ENTRENAR EL MODELO
fit!(mach, rows=train_rows)

# 5. HACER PREDICCIONES
# MLJ puede predecir la clase directamente con `predict_mode`
predicciones = MLJ.predict_mode(mach, rows=test_rows)

# 6. EVALUAR EL MODELO
# MLJ tiene métricas incorporadas
accuracy = accuracy(predicciones, y[test_rows])

println("Precisión (Accuracy) del modelo MLJ: ", round(accuracy * 100, digits=2), "%")