import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
import pandas as pd


def fit_linear_regression(X, y):
    # pad with 1's
    X = np.pad(X, [(0, 0), (1, 0)], constant_values=1)
    # Least square fit
    U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
    # compute X_pseudo_inverse
    X_pinv = VT.T @ np.linalg.inv(np.diag(Sigma)) @ U.T
    w = X_pinv @ y
    return w, Sigma


def predict(X, w):
    # pad X with 1's so we'll get w0 + a1w1 + a2w2 ...
    X = np.pad(X, [(0, 0), (1, 0)], constant_values=1)
    return X @ w


def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))


def load_data(path):
    data = pd.read_csv(path, na_values=['no info', '.']).copy()
    data.dropna(inplace=True)
    data.drop(['id', 'date', "long", "lat", "sqft_lot15"], axis='columns', inplace=True)
    data = data[data["price"] >= 0]
    data = data[data["bedrooms"] <= 15]
    data = data[data["yr_built"] >= 1900]
    data = data[[col for col in data.columns if col != 'price'] + ['price']]
    for category in ("floors",):
        data[category] = data[category].astype(float)
    for category in ("zipcode", ):
        labels = data[category].astype('category').cat.categories.tolist()
        replace_map_comp = {category: {k: v for k, v in zip(labels, list(range(1, len(labels) + 1)))}}
        data.replace(replace_map_comp, inplace=True)
        data[category] = data[category].astype('category')
    return data


def plot_singular_value(singular_values):
    x = np.arange(len(singular_values)) + 1
    y = sorted(singular_values, reverse=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'ro-', linewidth=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_title('singular values')
    ax.set_xlabel('singular values')
    ax.set_ylabel('value')
    leg = ax.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, shadow=False,
                    prop=font_manager.FontProperties(size='small'), markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    fig.savefig("Eigenvalues.png")
    fig.show()


def inspect_singular_values():
    dataset = load_data("kc_house_data.csv")
    dataset = dataset.values
    X = np.array(dataset[:, :-1], dtype='float')
    y = dataset[:, -1]
    _, singulars = fit_linear_regression(X, y)
    plot_singular_value(singulars)


def test_model_based_on_percentage():
    dataset = load_data("kc_house_data.csv")
    dataset = dataset.values
    np.random.shuffle(dataset)
    split_size = len(dataset) // 4
    train_set, test_set = dataset[split_size:, :], dataset[:split_size, :]
    X_train = np.array(train_set[:, :-1], dtype='float')
    y_train = np.array(train_set[:, -1], dtype='float')
    X_test = np.array(test_set[:, :-1], dtype='float')
    y_test = np.array(test_set[:, -1], dtype='float')
    training_scores = np.zeros(100, dtype=np.float64)
    p_s = np.arange(100) + 1
    for p in p_s:
        percentage = len(y_train) // (101 - p)
        if percentage == 100:
            percentage = -1
        w, _ = fit_linear_regression(X_train[:percentage], y_train[:percentage])
        y_hat = predict(X_test, w)
        training_scores[p - 1] = mse(y_test, y_hat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.plot(p_s, training_scores, 'ro-', linewidth=2)

    ax.set_title('training scores')
    ax.set_xlabel('percentage of dataset')
    ax.set_ylabel('mse')

    fig.savefig("16.png")
    fig.show()


def pearson_correlation(v1, v2):
    cov_v1_v2 = np.cov(v1, v2)
    return cov_v1_v2 / (np.std(v1) * np.std(v2))


def convert_df_to_np(df):
    return df.values


def feature_evaluation(X, y):
    y = y.values
    for col in X.select_dtypes(include=np.number):
        # plt.yscale("log")
        plt.scatter(X[col], y)
        correlation = pearson_correlation(X[col].values.reshape(1, -1), y.T)
        plt.title(col + f", correlation={round(correlation[0][-1], 3)}")
        plt.xlabel(col)
        plt.ylabel("costs")
        plt.show()


def inspect_correlation_of_between_features():
    dataset = load_data("kc_house_data.csv")
    feature_evaluation(dataset.iloc[:, :-1], dataset.iloc[:, -1:])


if __name__ == '__main__':
    inspect_singular_values()
    test_model_based_on_percentage()
    inspect_correlation_of_between_features()
