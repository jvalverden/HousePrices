from typing import Generator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import dython as dt

COLUMNS_TO_DROP = ['Id', 'FireplaceQu', 'Alley',
                   'PoolQC', 'Fence', 'MiscFeature', 'MiscVal']
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
N_SPLITS = 4


def get_categorical_indices(frame: pd.DataFrame) -> list:
    cols = frame.columns
    numerical_cols = frame.select_dtypes(include=np.number).columns
    categorical_columns = list(set(cols) - set(numerical_cols))
    return categorical_columns


def get_numerical_indices(frame: pd.DataFrame) -> list:
    return frame._get_numeric_data().columns


# IMPURE FOR SPEED!
def normalise_YN_fields(frame: pd.DataFrame) -> None:
    frame.replace(["Y", "N"], [1, 0], inplace=True)


def get_onehot_columns(frame: pd.DataFrame, categorical_indices: list) -> pd.DataFrame:
    one_hot_cols = []

    for col_name in categorical_indices:
        one_hot_cols.append(pd.get_dummies(frame.loc[:, col_name], prefix=col_name))

    new_columns = pd.concat(one_hot_cols, axis=1)
    return new_columns


def get_normalised_data(frame: pd.DataFrame, numerical_indices: list) -> pd.DataFrame:
    cols: pd.DataFrame = frame.loc[:, numerical_indices]

    return (cols - cols.min()) / (cols.max() - cols.min())


def get_standardized_data(frame: pd.DataFrame, numerical_indices: list) -> pd.DataFrame:
    cols: pd.DataFrame = frame.loc[:, numerical_indices]

    return (cols - cols.mean()) / cols.std()


def get_dataframe(columns_to_drop: list) -> pd.DataFrame:
    frame = pd.read_csv(TRAIN_FILENAME)
    # drop columns with NaN for too many instances
    frame = frame.drop(columns_to_drop, axis=1)

    return frame


def do_preprocessing(frame: pd.DataFrame, standardize: bool = False, normalize: bool = False):
    categorical_indices = get_categorical_indices(frame)
    numerical_indices = get_numerical_indices(frame)

    categorical_data = get_onehot_columns(frame, categorical_indices)

    if standardize:
        numerical_data = get_standardized_data(frame, numerical_indices)
    else:
        if normalize:
            numerical_data = get_normalised_data(frame, numerical_indices)
        else:
            numerical_data = frame.loc[:, numerical_indices]

    frame = pd.concat([numerical_data, categorical_data], axis="columns")
    frame.fillna(0, inplace=True)
    return frame


def get_mae(regression_model: LinearRegression, X_test: pd.Series, y_test: pd.Series, counter: int):
    y_unit = y_test.iat[0]
    y_pred = regression_model.predict(X_test)[0]

    mae = y_pred - y_unit

    if abs(mae) > 100000:
        print(counter, y_unit, y_pred)

    return mae


# calculate MES and RMSE and R^2 and MAE
def generate_matrics(regression_model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    y_pred = regression_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    num_feature = len(X_test.columns)

    print(
        f"Model {model_name}:\nNumber of features: {num_feature}\nMSE: {mse}\nRMSE: {rmse}\nR-squared: {r2}\nMAE: {mae}\n")


# plot regression
def plot_scatter(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    num_cols = X_train.shape[1]
    num_rows = (num_cols - 1) // 3 + 1

    _, axs = plt.subplots(nrows=num_rows, ncols=3,
                          figsize=(20, 2 * num_rows), sharey=True)

    for i, ax in enumerate(axs.flat):
        if i < num_cols:
            ax.scatter(X_train.iloc[:, i], y_train, alpha=0.5)
            ax.set_xlabel(X_train.columns[i])
            ax.set_ylabel('Sale Price')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# Create correlation matrix

def draw_correlation_matrix(frame: pd.DataFrame) -> None:
    corr = frame.corr()
    sns.heatmap(corr, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


def get_high_corr_list(X: pd.DataFrame, y: pd.Series, threshold: float) -> list:
    # create the list of features with corr_abs > threshold
    corr_abs = np.abs(X.corrwith(y))
    strong_corr = corr_abs[corr_abs > threshold]
    features = list(strong_corr.index)

    print(features)

    return features


def data_frame_split(data: pd.DataFrame | pd.Series, n_splits: int) -> list[pd.DataFrame | pd.Series]:
    result: list[pd.DataFrame | pd.Series] = []
    prev_split: int = 0

    for i in range(0, n_splits):
        split_idx: int = round(((i + 1) * (len(data) / n_splits)))
        result.append(data.iloc[prev_split:split_idx])
        prev_split = split_idx

    return result


# works like test_train_split, but instead returns a generator/iterator over all cross validation sets
def generate_cross_validation_sets(*data: pd.DataFrame | pd.Series, n_splits: int) -> Generator[list, None, None]:
    features = data_frame_split(data[0], n_splits)
    labels = data_frame_split(data[1], n_splits)

    for idx in range(n_splits):
        train_features = pd.concat(features[:idx] + features[idx + 1:])
        test_features = features[idx]
        train_labels = pd.concat(labels[:idx] + labels[idx + 1:])
        test_labels = labels[idx]

        yield list([train_features, test_features, train_labels, test_labels])

    return None


def run_linear_model(features: pd.DataFrame, labels: pd.Series, plot: bool, run_cross_val: bool,
                     model_name: str) -> None:
    if run_cross_val:
        errors = []
        counter = 1
        for X_train, X_test, y_train, y_test in generate_cross_validation_sets(features, labels, n_splits=N_SPLITS):
            model = LinearRegression()
            model.fit(X_train, y_train)

            if plot:
                # plot_scatter(X_train, y_train)
                draw_correlation_matrix(pd.concat([features, labels], axis="columns"))

        #     errors.append(get_mae(model, X_test, y_test, counter))
        #     counter += 1
        #
            generate_matrics(model, X_test, y_test, model_name)
        # plt.plot(errors)
        # plt.show()
        return

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    if plot:
        # plot_scatter(X_train, y_train)
        draw_correlation_matrix(pd.concat([features, labels], axis="columns"))

    generate_matrics(model, X_test, y_test, model_name)


def run_full_model(X: pd.DataFrame, y: pd.Series, plot: bool, run_cross_val: bool) -> None:
    run_linear_model(X, y, plot, run_cross_val, "full")


def run_reduced_model(X: pd.DataFrame, y: pd.Series, plot: bool, run_cross_val: bool, correlation: float) -> None:
    # values with low correlation
    features_to_drop = list(set(X.columns) - set(get_high_corr_list(X, y, correlation)))
    X = X.drop(columns=features_to_drop)

    run_linear_model(X, y, plot, run_cross_val, "reduced")


def main():
    train = get_dataframe([])
    train.drop(index=[325, 1298], inplace=True)  # this instance is a huge outlier
    normalise_YN_fields(train)
    labels = train.loc[:, "SalePrice"]
    features = train.drop(columns="SalePrice")

    features = do_preprocessing(features, standardize=True)

    print(features)

    run_reduced_model(features, labels, False, True, 0.5)


main()
