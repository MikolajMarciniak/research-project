from river import datasets
from river import compose
from river import linear_model
from river import metrics
from river import evaluate
from river import preprocessing
from river import feature_extraction
from river import stats
import itertools
import datetime as dt
import matplotlib.pyplot as plt

def get_hour(x):
    x['hour'] = x['moment'].hour
    return x

def evaluate_and_plot_curve(dataset, model, metric, delay_minutes, label, color):
    scores = evaluate.iter_progressive_val_score(
        dataset=dataset,
        model=model.clone(),
        metric=metric,
        moment='moment',
        delay=dt.timedelta(minutes=delay_minutes),
        step=20000
    )
    maes, step_counts = [], []
    for score in scores:
        maes.append(score["MAE"].get())
        step_counts.append(score["Step"])

    plt.plot(step_counts, maes, label=f'{label}: {delay_minutes} minutes', color=color)
    return


dataset = datasets.Bikes()

model = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
model += (
    get_hour |
    feature_extraction.TargetAgg(by=['station', 'hour'], how=stats.Mean())
)
model |= preprocessing.StandardScaler()
model |= linear_model.LinearRegression()

plt.figure(figsize=(10, 6))

# Evaluate and plot learning curve for 15 minutes
mean_score_15min = evaluate_and_plot_curve(dataset, model, metrics.MAE(), 15, 'MAE', color='blue')

# Evaluate and plot learning curve for 30 minutes
mean_score_30min = evaluate_and_plot_curve(dataset, model, metrics.MAE(), 30, 'MAE', color='green')

# Evaluate and plot learning curve for 60 minutes
mean_score_60min = evaluate_and_plot_curve(dataset, model, metrics.MAE(), 60, 'MAE', color='red')

plt.xlabel('Iterations')
plt.ylabel('Mean Absolute Error')
plt.title('Learning Curves')
plt.legend()
plt.savefig('learning_curves_combined.jpg')
plt.show()
