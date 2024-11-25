import streamlit as st
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import io
import joblib

# Load the dataset
filename = 'D:/school/COLLEGE/SUBJECTS/4TH YEAR/ITD105/LABEX03/LABEX03ITD105/DataSet/Rice_yield1.csv'
dataframe = pd.read_csv(filename)

# Automatically set the last column as the target variable
target_column = dataframe.columns[-1]  # Get the last column
feature_columns = dataframe.columns[:-1]  # Get all columns except the last one

# Dynamically select X and Y based on the selections
X = dataframe[feature_columns].values
Y = dataframe[target_column].values

# List of algorithms for selection
algorithms = ['Decision Tree Regressor', 'Elastic Net', 'AdaBoost Regressor', 
              'K-Nearest Neighbors', 'Lasso Regression', 'Ridge Regression', 
              'Linear Regression', 'MLP Regressor', 'Random Forest Regressor', 
              'Support Vector Regressor (SVR)']

# Create an empty dataframe to store accuracy values
accuracy_df = DataFrame(columns=["ML Algorithm (Classification)", "Mean Squared Error (MSE)"])
# Create an empty dataframe to store hyperparameter values
hyperparam_df = DataFrame(columns=["ML Algorithm", "Hyperparameters"])

# Function to append accuracy to the dataframe
def add_accuracy_to_df(algorithm, mse):
    global accuracy_df
    mse_value = -mse.mean()  # Convert negative MSE to positive
    accuracy_df = pd.concat([accuracy_df, DataFrame([{"ML Algorithm (Classification)": algorithm, "Mean Squared Error (MSE)": "%.3f" % mse_value}])], ignore_index=True)

# Function to append hyperparameters to the dataframe
def add_hyperparameters_to_df(algorithm, hyperparameters):
    global hyperparam_df
    hyperparam_df = pd.concat([hyperparam_df, DataFrame([{"ML Algorithm": algorithm, "Hyperparameters": hyperparameters}])], ignore_index=True)

# Dictionary to store models
models = {}

# Algorithm selection and model training
for selected_algo in algorithms:
    mse = None
    if selected_algo == 'Decision Tree Regressor':
        with st.sidebar.expander("Decision Tree Regressor Hyperparameters", expanded=False):
            max_depth = st.slider("Max Depth", 1, 20, 1, key=f"max_depth_{selected_algo}")
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2, key=f"min_samples_split{selected_algo}")
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1, key=f"min_samples_leaf{selected_algo}")
            n_splits = st.slider("Number of Folds (K)", 2, 20, 10, key=f"n_splits{selected_algo}")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if max_depth != 1 or min_samples_split != 2 or min_samples_leaf != 1:
            add_hyperparameters_to_df(selected_algo, f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}")

    elif selected_algo == 'Elastic Net':
        with st.sidebar.expander("Elastic Net Hyperparameters", expanded=False):
            alpha = st.slider("Alpha (Regularization Strength)", 0.0, 5.0, 1.0, 0.1, key=f"alpha{selected_algo}")
            l1_ratio = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.01, key=f"l1_ratio{selected_algo}")
            max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100, key=f"max_iter{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=None)
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if alpha != 1.0 or l1_ratio != 0.5 or max_iter != 1000:
            add_hyperparameters_to_df(selected_algo, f"Alpha: {alpha}, L1 Ratio: {l1_ratio}, Max Iterations: {max_iter}")

    elif selected_algo == 'AdaBoost Regressor':
        with st.sidebar.expander("AdaBoost Regressor Hyperparameters", expanded=False):
            n_estimators = st.slider("Number of Estimators", 1, 200, 50, 1, key=f"n_estimators{selected_algo}")
            learning_rate = st.slider("Learning Rate", 0.01, 5.0, 1.0, 0.01, key=f"learning_rate{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        
        ada_model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=None)
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(ada_model, X, Y, cv=kfold, scoring=scoring)
        ada_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = ada_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if n_estimators != 50 or learning_rate != 1.0:
            add_hyperparameters_to_df(selected_algo, f"Number of Estimators: {n_estimators}, Learning Rate: {learning_rate}")

    elif selected_algo == 'K-Nearest Neighbors':
        with st.sidebar.expander("K-Nearest Neighbors Hyperparameters", expanded=False):
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1, key=f"n_neighbors{selected_algo}")
            weights = st.selectbox("Weights", ["uniform", "distance"], key=f"weights{selected_algo}")
            algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"], key=f"algorithm{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        knn_model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(knn_model, X, Y, cv=kfold, scoring=scoring)
        knn_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = knn_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if n_neighbors != 5 or weights != "uniform" or algorithm != "auto":
            add_hyperparameters_to_df(selected_algo, f"Number of Neighbors: {n_neighbors}, Weights: {weights}, Algorithm: {algorithm}")

    elif selected_algo == 'Lasso Regression':
        with st.sidebar.expander("Lasso Regression Hyperparameters", expanded=False):
            alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key=f"alpha{selected_algo}")
            max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key=f"max_iter{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        lasso_model = Lasso(alpha=alpha, max_iter=max_iter, random_state=None)
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(lasso_model, X, Y, cv=kfold, scoring=scoring)
        lasso_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = lasso_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if alpha != 1.0 or max_iter != 1000:
            add_hyperparameters_to_df(selected_algo, f"Alpha: {alpha}, Max Iterations: {max_iter}")

    elif selected_algo == 'Ridge Regression':
        with st.sidebar.expander("Ridge Regression Hyperparameters", expanded=False):
            alpha = st.slider("Regularization Parameter (alpha)", 0.01, 10.0, 1.0, 0.01, key=f"alpha{selected_algo}")
            max_iter = st.slider("Maximum Iterations", 100, 1000, 1000, 100, key=f"max_iter{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        ridge_model = Ridge(alpha=alpha, max_iter=max_iter, random_state=None)
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(ridge_model, X, Y, cv=kfold, scoring=scoring)
        ridge_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = ridge_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if alpha != 1.0 or max_iter != 1000:
            add_hyperparameters_to_df(selected_algo, f"Alpha: {alpha}, Max Iterations: {max_iter}")

    elif selected_algo == 'Linear Regression':
        with st.sidebar.expander("Linear Regression Hyperparameters", expanded=False):
            st.write("N/A")

        kfold = KFold(n_splits=10, random_state=None)
        model = LinearRegression()
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = model

    elif selected_algo == 'MLP Regressor':
        with st.sidebar.expander("MLP Regressor Hyperparameters", expanded=False):
            hidden_layer_sizes = st.slider("Hidden Layer Sizes", min_value=10, max_value=200, value=(100, 50), step=10, key=f"hidden_layer_sizes{selected_algo}")
            activation = st.selectbox("Activation Function", options=['identity', 'logistic', 'tanh', 'relu'], index=3, key=f"activation{selected_algo}")
            solver = st.selectbox("Solver", options=['adam', 'lbfgs', 'sgd'], index=0, key=f"solver{selected_algo}")
            learning_rate = st.selectbox("Learning Rate Schedule", options=['constant', 'invscaling', 'adaptive'], index=0, key=f"learning_rate{selected_algo}")
            max_iter = st.slider("Max Iterations", min_value=100, max_value=2000, value=1000, step=100, key=f"max1{selected_algo}")
            random_state = st.number_input("Random State", value=50, key=f"random_state{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        mlp_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state
        )
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(mlp_model, X, Y, cv=kfold, scoring=scoring)
        mlp_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = mlp_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if hidden_layer_sizes != (100, 50) or activation != 'relu' or solver != 'adam' or learning_rate != 'constant' or max_iter != 1000 or random_state != 50:
            add_hyperparameters_to_df(selected_algo, f"Hidden Layer Sizes: {hidden_layer_sizes}, Activation: {activation}, Solver: {solver}, Learning Rate: {learning_rate}, Max Iterations: {max_iter}, Random State: {random_state}")

    elif selected_algo == 'Random Forest Regressor':
        with st.sidebar.expander("Random Forest Regressor Hyperparameters", expanded=False):
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100, step=10, key=f"n_estimators{selected_algo}")
            max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=None, key=f"max_depth{selected_algo}")
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, key=f"min_samples_split{selected_algo}")
            min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, key=f"min_samples_leaf{selected_algo}")
            random_state = st.number_input("Random State", value=42, key=f"random_state{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(rf_model, X, Y, cv=kfold, scoring=scoring)
        rf_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = rf_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if n_estimators != 100 or max_depth is not None or min_samples_split != 2 or min_samples_leaf != 1 or random_state != 42:
            add_hyperparameters_to_df(selected_algo, f"Number of Trees: {n_estimators}, Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}, Random State: {random_state}")

    elif selected_algo == 'Support Vector Regressor (SVR)':
        with st.sidebar.expander("Support Vector Regressor (SVR) Hyperparameters", expanded=False):
            kernel = st.selectbox("Kernel", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2, key=f"kernel{selected_algo}")
            C = st.slider("Regularization Parameter (C)", min_value=0.01, max_value=100.0, value=1.0, step=0.01, key=f"C{selected_algo}")
            epsilon = st.slider("Epsilon", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key=f"epsilon{selected_algo}")

        kfold = KFold(n_splits=10, random_state=None)
        svm_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        scoring = 'neg_mean_squared_error'
        mse = cross_val_score(svm_model, X, Y, cv=kfold, scoring=scoring)
        svm_model.fit(X, Y)  # Fit the model to the entire dataset
        models[selected_algo] = svm_model
        
        # Add hyperparameters to the summary DataFrame if they are edited
        if kernel != 'rbf' or C != 1.0 or epsilon != 0.1:
            add_hyperparameters_to_df(selected_algo, f"Kernel: {kernel}, C: {C}, Epsilon: {epsilon}")

    # Add the accuracy to the DataFrame
    add_accuracy_to_df(selected_algo, mse)

# Display the table with all algorithm accuracies
st.title("ML Algorithm Table")
if not accuracy_df.empty:  # Ensure the DataFrame is not empty
    accuracy_df["Mean Squared Error (MSE)"] = pd.to_numeric(accuracy_df["Mean Squared Error (MSE)"], errors='coerce')
    max_mse = accuracy_df["Mean Squared Error (MSE)"].max()
    min_mse = accuracy_df["Mean Squared Error (MSE)"].min()

    def highlight_row(row):
        if row["Mean Squared Error (MSE)"] == min_mse:
            return ['color: green' if col == "Mean Squared Error (MSE)" else '' for col in row.index]
        elif row["Mean Squared Error (MSE)"] == max_mse:
            return ['color: red' if col == "Mean Squared Error (MSE)" else '' for col in row.index]
        else:
            return ['' for _ in row.index]

    styled_df = accuracy_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df)

# Display the hyperparameter summary table
st.title("Hyperparameter Summary Table")
if not hyperparam_df.empty:  # Ensure the DataFrame is not empty
    st.dataframe(hyperparam_df)

# Plotting Bar graph for all algorithms
st.title("ML Algorithm Bar Graph")
st.bar_chart(accuracy_df.set_index("ML Algorithm (Classification)")['Mean Squared Error (MSE)'])

# Dropdown to select the model for download
selected_algo = st.selectbox(
    "Select the algorithm model to download:",
    options=algorithms,
    index=0  # Set default selection
)

# Dropdown to select model for download
model_to_download = models.get(selected_algo)

# For each selected algorithm, save the model in memory and allow download
if model_to_download:
    model_buffer = io.BytesIO()
    joblib.dump(model_to_download, model_buffer)
    model_buffer.seek(0)
    
    st.download_button(
        label=f"Download {selected_algo} Model",
        data=model_buffer,
        file_name=f"{selected_algo}_model.joblib",
        mime="application/octet-stream"
    )