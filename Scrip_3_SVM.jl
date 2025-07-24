using MLJ, DataFrames, LIBSVM, Statistics

# 1. INSTALAR Y CARGAR PAQUETES
# using Pkg; Pkg.add("LIBSVM")
# --------------------------------------------------------------------
# Cargamos el modelo SVC (Support Vector Classifier) desde el paquete
SVC = @load SVC pkg=LIBSVM

# 2. GENERAR DATOS DE EJEMPLO
# Crearemos dos "nubes" de puntos que no son perfectamente separables por una línea recta
function generar_datos_svm(n_puntos_por_clase=100)
    # Clase 1
    X1_clase1 = randn(n_puntos_por_clase) .+ 2
    X2_clase1 = randn(n_puntos_por_clase) .+ 2
    
    # Clase 2
    X1_clase2 = randn(n_puntos_por_clase) .- 2
    X2_clase2 = randn(n_puntos_por_clase) .- 2
    
    # Unir datos y etiquetas
    X1 = vcat(X1_clase1, X1_clase2)
    X2 = vcat(X2_clase1, X2_clase2)
    y = vcat(fill("Clase A", n_puntos_por_clase), fill("Clase B", n_puntos_por_clase))
    
    return DataFrame(X1=X1, X2=X2), y
end

X, y_raw = generar_datos_svm()

# 3. PREPARAR DATOS PARA MLJ
# Convertimos el vector de etiquetas a tipo Categorial
y = categorical(y_raw)

# 4. DIVIDIR DATOS Y ENTRENAR EL MODELO
# Usamos el flujo de trabajo estándar de MLJ
(train_rows, test_rows) = partition(eachindex(y), 0.8, shuffle=true)

# Instanciamos el modelo y lo envolvemos en una máquina
svm_model = SVC()
svm_machine = machine(svm_model, X, y)

# Entrenamos la máquina solo con los datos de entrenamiento
fit!(svm_machine, rows=train_rows)

# 5. PREDECIR Y EVALUAR
# Asegúrate de pasar la lista completa `test_rows`
predicciones = MLJ.predict_mode(svm_machine, rows=test_rows)

# Ahora `accuracy` recibe una lista, como es debido
accuracy_score = accuracy(predicciones, y[test_rows])

println("Precisión (Accuracy) del modelo SVM: ", round(accuracy_score * 100, digits=2), "%")