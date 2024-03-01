from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from river import preprocessing as p
from river import linear_model as l
from river import compat
from river import compose

# Load the data
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

# Define the steps of the model
model = pipeline.Pipeline([
    ('scale', preprocessing.StandardScaler()),
    ('lin_reg', linear_model.LogisticRegression(solver='lbfgs'))
])

# Define a determistic cross-validation procedure
cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# Compute the MSE values
scorer = metrics.make_scorer(metrics.roc_auc_score)
scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# Display the average score and it's standard deviation
print(f'Batch ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')

# We define a Pipeline, exactly like we did earlier for sklearn 
model = compose.Pipeline(
    ('scale', p.StandardScaler()),
    ('log_reg', l.LogisticRegression())
)

# We make the Pipeline compatible with sklearn
model = compat.convert_river_to_sklearn(model)

# We compute the CV scores using the same CV scheme and the same scoring
scores = model_selection.cross_val_score(model, X, y, scoring=scorer, cv=cv)

# Display the average score and it's standard deviation
print(f'Online ROC AUC: {scores.mean():.3f} (± {scores.std():.3f})')
