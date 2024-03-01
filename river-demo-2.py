from pprint import pprint
from river import datasets
from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import optim
from river import feature_extraction
from river import stats
import itertools
import datetime as dt

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

dataset = datasets.Bikes()

for x, y in dataset:
    pprint(x)
    print(f'Number of available bikes: {y}')
    break

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

for x, y in itertools.islice(dataset, 10000):
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
    
x, y = next(iter(dataset))
print(model.debug_one(x))

metric = metrics.MAE()


evaluate.progressive_val_score(
    dataset=dataset,
    model=model.clone(),
    metric=metrics.MAE(),
    moment='moment',
    delay=dt.timedelta(minutes=30),
    print_every=20_000
)