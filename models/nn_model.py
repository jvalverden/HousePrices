import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time
import seaborn as sns
from datetime import datetime as dt
from models.Regression import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def get_embeds(frame: pd.DataFrame, categorical_indices: list, numerical_indices: list):
    y = ['SalePrice']
    for cat in categorical_indices:
        frame[cat] = frame[cat].astype('category')

    cats = np.stack([frame[col].cat.codes.values for col in categorical_indices], 1)
    cats = torch.tensor(cats, dtype=torch.int64)
    # Convert continuous variables to a tensorframe
    conts = np.stack([frame[col].values for col in numerical_indices], 1)
    conts = torch.tensor(conts, dtype=torch.float)
    # Create outcome
    y = torch.tensor(frame[y].values, dtype=torch.float).reshape(-1, 1)
    # Set embedding sizes
    cat_szs = [len(frame[col].cat.categories) for col in categorical_indices]
    emb_szs = [(size, min(50, (size + 1) // 2)) for size in cat_szs]

    return [emb_szs, conts, cats]


def run_nn():
    from ML_final import get_dataframe, COLUMNS_TO_DROP, TRAIN_FILENAME
    data_name = 'housing_price'
    y = ['SalePrice']

    train, [emb_szs, conts, cats] = get_dataframe(COLUMNS_TO_DROP, TRAIN_FILENAME, use_nn=True)

    print(emb_szs, conts, cats)

    # Divide obs in half to get half batch size
    batch_size = len(train.columns) // 2

    torch.manual_seed(123)
    model = MLPRegressor(emb_szs, conts.shape[1], out_sz=1, layers=[200, 100], p=0.4)
    print('[INFO] Model definition')
    print(model)
    print('=' * 80)
    # split the data
    test_size = int(batch_size * .3)
    cat_train = cats[:batch_size - test_size]
    cat_test = cats[batch_size - test_size:batch_size]
    con_train = conts[:batch_size - test_size]
    con_test = conts[batch_size - test_size:batch_size]
    y_train = y[:batch_size - test_size]
    y_test = y[batch_size - test_size:batch_size]

    # Train model

    def train(model, y_train, categorical_train, continuous_train,
              y_val, categorical_valid, continuous_valid,
              learning_rate=0.001, epochs=300, print_out_interval=2):
        global criterion
        criterion = nn.MSELoss()  # we'll convert this to RMSE later
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        start_time = time.time()
        model.train()

        losses = []
        preds = []

        for i in range(epochs):
            i += 1  # Zero indexing trick to start the print out at epoch 1
            y_pred = model(categorical_train, continuous_train)
            preds.append(y_pred)
            loss = torch.sqrt(criterion(y_pred, y_train))  # RMSE
            losses.append(loss)

            if i % print_out_interval == 1:
                print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('=' * 80)
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')  # print the last line
        print(f'Duration: {time.time() - start_time:.0f} seconds')  # print the time elapsed

        # Evaluate model
        with torch.no_grad():
            y_val = model(categorical_valid, continuous_valid)
            mse = criterion(y_val, y_test)
            rmse = torch.sqrt(criterion(y_val, y_test))
            r2 = r2_score(y_test.cpu().numpy(), y_val.cpu().numpy())
            mae = F.l1_loss(y_val, y_test)
            num_feature = len(cat_test) + len(con_test)
        print(
            f'Number of features: {num_feature}\nMSE: {mse}\nRMSE: {rmse}\nR-squared: {r2}\nMAE: {mae}\nRMSE: {loss:.8f}')

        # Create empty list to store my results
        preds = []
        diffs = []
        actuals = []

        for i in range(len(categorical_valid)):
            diff = np.abs(y_val[i].item() - y_test[i].item())
            pred = y_val[i].item()
            actual = y_test[i].item()

            diffs.append(diff)
            preds.append(pred)
            actuals.append(actual)

        valid_results_dict = {
            'predictions': preds,
            'diffs': diffs,
            'actuals': actuals
        }

        # Save model
        torch.save(model.state_dict(), f'model_artifacts/{data_name}_{epochs}.pt')
        # Return components to use later
        return losses, preds, diffs, actuals, model, valid_results_dict, epochs

    losses, preds, diffs, actuals, model, valid_results_dict, epochs = train(
        model=model, y_train=y_train,
        categorical_train=cat_train,
        continuous_train=con_train,
        y_val=y_test,
        categorical_valid=cat_test,
        continuous_valid=con_test,
        learning_rate=0.01,
        epochs=400,
        print_out_interval=25)
    valid_res = pd.DataFrame(valid_results_dict)

    # Visualise results
    current_time = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.figure()
    sns.scatterplot(data=valid_res,
                    x='predictions', y='actuals', size='diffs', hue='diffs')
    plt.title('Validation Result')
    plt.show()
    # Produce validation graph
    losses_collapsed = [losses[i].item() for i in range(epochs)]
    epochs = [ep + 1 for ep in range(epochs)]
    eval_df = pd.DataFrame({
        'epochs': epochs,
        'loss': losses_collapsed
    })

    plt.figure()
    palette = sns.color_palette("mako_r", 6)
    sns.lineplot(data=eval_df, x='epochs', y='loss', palette=palette)
    plt.title('Model training graph')
    plt.show()
    emb_szs1 = [(4, 2), (3, 2), (5, 3), (1, 1), (5, 3), (7, 4), (5, 3), (16, 8), (3, 2), (4, 2), (6, 3), (5, 3), (6, 3),
                (6, 3), (2, 1), (4, 2), (6, 3), (5, 3), (4, 2), (5, 3), (7, 4), (5, 3), (9, 5), (5, 3), (4, 2), (4, 2),
                (5, 3), (6, 3), (4, 2), (2, 1), (5, 3), (8, 4), (4, 2), (5, 3), (25, 13), (5, 3), (5, 3), (3, 2),
                (14, 7),
                (3, 2), (9, 5), (7, 4), (6, 3)]
    torch.manual_seed(123)
    model = MLPRegressor(emb_szs1, conts.shape[1], out_sz=1, layers=[200, 100], p=0.4)
    print('[INFO] Model definition')
    print(model)
    print('=' * 80)
    model_infer = MLPRegressor(emb_szs1, conts.shape[1], 1, [200, 100], p=0.4)
    model_infer.load_state_dict(torch.load('../model_artifacts/housing_price_400.pt'))
    print(model_infer.eval())

    def prod_data(model, cat_prod, cont_prod, verbose=False):
        # Pass the inputs from the cont and cat tensors to function
        with torch.no_grad():
            y_val = model(cat_prod, cont_prod)

        # Get preds on prod data
        preds = []
        for i in range(len(cat_prod)):
            result = y_val[i].item()
            preds.append(result)

            if verbose:
                print(f'The predicted value is: {y_val[i].item()}')

        return preds

    prod = prod_data(model_infer, cats, conts, verbose=True)
    # Print out prod
    print(len(prod))
    plt.scatter(y.numpy(), prod)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Predicted values and Actual values ')
    plt.show()
