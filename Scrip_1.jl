
# Install Pkg.add("MLDatasets") to use data
using Pkg
#Pkg.add("MLDatasets")
using MLDatasets

using MLJ

# small mincer data
import DataFrames: DataFrame
X = DataFrame(
    name = ["Siri", "Robo", "Alexa", "Cortana"],
    gender = ["male", "male", "Female", "female"],
    likes_soup = [true, false, false, true],
    height = [152, missing, 148, 163],
    rating = [2, 5, 2, 1],
    outcome = ["rejected", "accepted", "accepted", "rejected"],
)
schema(X)
# working with categorical data
X.outcome = coerce(X.outcome, OrderedFactor)
levels(X.outcome)

Xnew = coerce(X, :gender     => Multiclass,
                 :likes_soup => OrderedFactor,
                 :height     => Continuous,
                 :rating     => OrderedFactor)
schema(Xnew)