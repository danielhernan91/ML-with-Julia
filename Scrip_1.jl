# ------------------------------
# PAQUETES NECESARIOS
# ------------------------------
using MLJ
using MLJLinearModels
using DataFrames
using ScientificTypes
using StatsBase
using Random
using Plots

# ------------------------------
# CREACIÓN DEL DATAFRAME
# ------------------------------
data = DataFrame(
    id = ["01", "02", "03", "04"],
    gender = ["male", "male", "female", "female"],
    job = [true, true, true, true],
    income = [1152, 1345, 1148, 1163],
    age = [22, 51, 24, 41],
    status = ["single", "married", "single", "single"],
)

# ------------------------------
# COERCIÓN DE TIPOS
# ------------------------------
schema(data)  # Ver tipos originales

data = coerce(data,
    :gender => Multiclass,
    :job => OrderedFactor,
    :income => Continuous,
    :age => Continuous,
    :status => OrderedFactor,
)

# ------------------------------
# SEPARACIÓN EN FEATURES Y TARGET
# ------------------------------
select!(data, Not(:id))  # Quitar columna 'id'
y = data.income
X = select(data, Not(:income))

# ------------------------------
# PARTICIÓN TRAIN/TEST
# ------------------------------
rng = MersenneTwister(123)
train, test = partition(eachindex(y), 0.75, shuffle=true, rng=rng)
X_train, X_test = X[train, :], X[test, :]
y_train, y_test = y[train], y[test]

# ------------------------------
# CARGA DE MODELOS
# ------------------------------
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0
OneHotEncoder = @load OneHotEncoder pkg=MLJModels verbosity=0
Standardizer = @load Standardizer pkg=MLJModels verbosity=0

# ------------------------------
# DEFINICIÓN DEL PIPELINE
# ------------------------------
@pipeline ModeloRegresion(
    one_hot = OneHotEncoder(drop_last=true),
    std = Standardizer(),
    model = LinearRegressor()
)

# ------------------------------
# ENTRENAMIENTO Y PREDICCIÓN
# ------------------------------
mach = machine(ModeloRegresion(), X_train, y_train)
fit!(mach)

y_pred = predict(mach, X_test)  # Produce un vector de UnivariateFinite o valores continuos

# ------------------------------
# EVALUACIÓN
# ------------------------------
using MLJBase: rms, mae, r²

println("Evaluación del modelo:")
println("RMSE: ", rms(y_pred, y_test))
println("MAE: ", mae(y_pred, y_test))
println("R²: ", r²(y_pred, y_test))

# ------------------------------
# VISUALIZACIÓN (opcional)
# ------------------------------
scatter(y_test, y_pred,
    xlabel = "Valor real",
    ylabel = "Predicción",
    title = "Predicciones vs Valores reales",
    label = "Predicciones",
    legend = :topleft)
plot!(identity, label = "Ideal", lw=2)
