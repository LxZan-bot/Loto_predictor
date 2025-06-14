import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from predictor import draw_to_vector

def train_model(csv_path='loto_data.csv'):
    df = pd.read_csv(csv_path)
    past_draws = df.iloc[:, 1:7].values.tolist()

    X, y = [], []
    for i in range(len(past_draws) - 1):
        vec = draw_to_vector(past_draws[i])
        next_draw = past_draws[i + 1]
        for n in range(1, 44):
            X.append(vec)
            y.append(1 if n in next_draw else 0)

    model = RandomForestClassifier()
    model.fit(X, y)
    return model
