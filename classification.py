import streamlit as st
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import io
import joblib

# Load the dataset
filename = 'D:/school/COLLEGE/SUBJECTS/4TH YEAR/ITD105/LABEX03/LABEX03ITD105/DataSet/heart1.csv'
dataframe = pd.read_csv(filename)

# Automatically set the last column as the target variable
target_column = dataframe.columns[-1]
feature_columns = dataframe.columns[:-1]

# Prepare data
X = dataframe[feature_columns].values
Y = dataframe[target_column].values

# List of algorithms for selection
algorithms = ['Decision Tree', 'Gaussian Naive Bayes', 'AdaBoost', 'K-Nearest Neighbors', 
              'Logistic Regression', 'MLP Classifier', 'Perceptron', 'Random Forest', 'Support Vector Machine']

# Create an empty dataframe to store accuracy values
accuracy_df = DataFrame(columns=["ML Algorithm (Classification)", "ROC AUC"])
hyperparam_df = DataFrame(columns=["ML Algorithm", "Hyperparameters"])

# Function to append accuracy to the dataframe
def add_accuracy_to_df(algorithm, roc_auc):
    global accuracy_df
    accuracy_df = pd.concat([accuracy_df, DataFrame([{"ML Algorithm (Classification)": algorithm, "ROC AUC": roc_auc}])], ignore_index=True)

# Function to append hyperparameters to the dataframe
def add_hyperparameters_to_df(algorithm, hyperparameters):
    global hyperparam_df
    hyperparam_df = pd.concat([hyperparam_df, DataFrame([{"ML Algorithm": algorithm, "Hyperparameters": hyperparameters}])], ignore_index=True)

# Dictionary to store models
models = {}

# Algorithm selection and model training
for selected_algo in algorithms:
    roc_auc = None
    if selected_algo == 'Decision Tree':
        # Default values
        default_test_size = 0.2
        default_random_seed = 50
        default_max_depth = 5
        default_min_samples_split = 2
        default_min_samples_leaf = 1
        
        with st.sidebar.expander("Decision Tree Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test_size_{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"random_seed_{selected_algo}")
            max_depth = st.slider("Max Depth", 1, 20, default_max_depth, key=f"max_depth_{selected_algo}")
            min_samples_split = st.slider("Min Samples Split", 2, 10, default_min_samples_split, key=f"min_samples_split_{selected_algo}")
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, default_min_samples_leaf, key=f"min_samples_leaf_{selected_algo}")

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=random_seed)
        model.fit(X_train, Y_train)

        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            max_depth != default_max_depth or 
            min_samples_split != default_min_samples_split or 
            min_samples_leaf != default_min_samples_leaf):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}")

    elif selected_algo == 'Gaussian Naive Bayes':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_var_smoothing = -9
        
        with st.sidebar.expander("Gaussian Naive Bayes Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test_size_{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"random_seed_{selected_algo}")
            var_smoothing = st.number_input("Var Smoothing (Log Scale)", min_value=-15, max_value=-1, value=default_var_smoothing, step=1, key=f"var_smoothing_{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)
        
        var_smoothing_value = 10 ** var_smoothing
        model = GaussianNB(var_smoothing=var_smoothing_value)
        model.fit(X_train, Y_train)

        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            var_smoothing != default_var_smoothing):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Var Smoothing: {var_smoothing_value}")

    elif selected_algo == 'AdaBoost':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_n_estimators = 50
        
        with st.sidebar.expander("AdaBoost Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test_size2_{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"random_seed2_{selected_algo}")
            n_estimators = st.slider("Number of Estimators", 1, 100, default_n_estimators, key=f"n_estimators_{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_seed)
        model.fit(X_train, Y_train)

        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            n_estimators != default_n_estimators):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Number of Estimators: {n_estimators}")

    elif selected_algo == 'K-Nearest Neighbors':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_n_neighbors = 5
        default_weights = "uniform"
        default_algorithm = "auto"
        
        with st.sidebar.expander("K-Nearest Neighbors Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"keytest3{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"seed3{selected_algo}")
            n_neighbors = st.slider("Number of Neighbors", 1, 20, default_n_neighbors, key=f"n_neighbors_{selected_algo}")
            weights = st.selectbox("Weights", options=["uniform", "distance"], index=0, key=f"weights_{selected_algo}")
            algorithm = st.selectbox("Algorithm", options=["auto", "ball_tree", "kd_tree", "brute"], index=0, key=f"algorithm_{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
        model.fit(X_train, Y_train)
        
        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            n_neighbors != default_n_neighbors or 
            weights != default_weights or 
            algorithm != default_algorithm):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Number of Neighbors: {n_neighbors}, Weights: {weights}, Algorithm: {algorithm}")

    elif selected_algo == 'Logistic Regression':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_max_iter = 200
        default_solver = "lbfgs"
        default_C = 1.0
        
        with st.sidebar.expander("Logistic Regression Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test4{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"seed4{selected_algo}")
            max_iter = st.slider("Max Iterations", 100, 500, default_max_iter, key=f"max_iter_{selected_algo}")
            solver = st.selectbox("Solver", options=["lbfgs", "liblinear", "sag", "saga", "newton-cg"], index=0, key=f"solver_{selected_algo}")
            C = st.number_input("Inverse of Regularization Strength", min_value=0.01, max_value=10.0, value=default_C, key=f"C_{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        model = LogisticRegression(max_iter=max_iter, solver=solver, C=C)
        model.fit(X_train, Y_train)

        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            max_iter != default_max_iter or 
            solver != default_solver or 
            C != default_C):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Max Iterations: {max_iter}, Solver: {solver}, C: {C}")

    elif selected_algo == 'MLP Classifier':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_hidden_layer_sizes = "65,32"
        default_activation = "relu"
        default_max_iter = 200
        
        with st.sidebar.expander("MLP Classifier Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test5{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"seed5{selected_algo}")
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (e.g., 65,32)", default_hidden_layer_sizes, key=f"hidden_layer_sizes_{selected_algo}")
            activation = st.selectbox("Activation Function", options=["identity", "logistic", "tanh", "relu"], index=3, key=f"activation_{selected_algo}")
            max_iter = st.slider("Max Iterations", 100, 500, default_max_iter, key=f"max5{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        hidden_layer_sizes = tuple(map(int, hidden_layer_sizes.split(',')))
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver='adam', max_iter=max_iter, random_state=random_seed)
        model.fit(X_train, Y_train)

        Y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            hidden_layer_sizes != tuple(map(int, default_hidden_layer_sizes.split(','))) or 
            activation != default_activation or 
            max_iter != default_max_iter):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Hidden Layer Sizes: {hidden_layer_sizes}, Activation: {activation}, Max Iterations: {max_iter}")

    elif selected_algo == 'Perceptron':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_max_iter = 200
        default_eta0 = 1.0
        default_tol = 1e-3
        
        with st.sidebar.expander("Perceptron Classifier Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test6{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"seed6{selected_algo}")
            max_iter = st.slider("Max Iterations", 100, 500, default_max_iter, key=f"max6{selected_algo}")
            eta0 = st.number_input("Initial Learning Rate", min_value=0.001, max_value=10.0, value=default_eta0, key=f"eta0_{selected_algo}")
            tol = st.number_input("Tolerance for Stopping Criterion", min_value=0.0001, max_value=1.0, value=default_tol, key=f"tol_{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        model = Perceptron(max_iter=max_iter, random_state=random_seed, eta0=eta0, tol=tol)
        model.fit(X_train, Y_train)

        Y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(Y_test, Y_scores)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            max_iter != default_max_iter or 
            eta0 != default_eta0 or 
            tol != default_tol):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Max Iterations: {max_iter}, Initial Learning Rate: {eta0}, Tolerance: {tol}")

    elif selected_algo == 'Random Forest':
        # Default values
        default_test_size = 0.2
        default_random_seed = 7
        default_n_estimators = 100
        default_max_depth = None  # None means no limit
        default_min_samples_split = 2
        default_min_samples_leaf = 1
    
        with st.sidebar.expander("Random Forest Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test7{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"seed7{selected_algo}")
            n_estimators = st.slider("Number of Estimators (Trees)", 10, 200, default_n_estimators, key=f"n_estimators{selected_algo}")
            max_depth = st.slider("Max Depth of Trees", 1, 50, default_max_depth, key=f"max_depth{selected_algo}")  # Allows None for no limit
            min_samples_split = st.slider("Min Samples to Split a Node", 2, 10, default_min_samples_split, key=f"min_samples_split{selected_algo}")
            min_samples_leaf = st.slider("Min Samples in Leaf Node", 1, 10, default_min_samples_leaf, key=f"min_samples_leaf{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)
        
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        model.fit(X_train, Y_train)
        
        Y_scores = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_scores)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            n_estimators != default_n_estimators or 
            max_depth != default_max_depth or 
            min_samples_split != default_min_samples_split or 
            min_samples_leaf != default_min_samples_leaf):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, Number of Estimators: {n_estimators}, Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Min Samples Leaf: {min_samples_leaf}")

    elif selected_algo == 'Support Vector Machine':
        # Default values
        default_test_size = 0.2
        default_random_seed = 42
        default_C = 1.0
        default_kernel = 'rbf'
        
        with st.sidebar.expander("Support Vector Machine (SVM) Hyperparameters", expanded=False):
            test_size = st.slider("Test Size (fraction)", 0.1, 0.5, default_test_size, key=f"test8{selected_algo}")
            random_seed = st.slider("Random Seed", 1, 100, default_random_seed, key=f"seed8{selected_algo}")
            C = st.slider("Regularization Parameter (C)", 0.1, 10.0, default_C, key=f"C{selected_algo}")
            kernel = st.selectbox("Kernel Type", options=['linear', 'poly', 'rbf', 'sigmoid'], index=2, key=f"kernel{selected_algo}")

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_seed)

        model = SVC(kernel=kernel, C=C, random_state=random_seed)
        model.fit(X_train, Y_train)
        
        Y_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(Y_test, Y_scores)
        models[selected_algo] = model
        
        # Check for changes and add to summary
        if (test_size != default_test_size or 
            random_seed != default_random_seed or 
            C != default_C or 
            kernel != default_kernel):
            add_hyperparameters_to_df(selected_algo, f"Test Size: {test_size}, Random Seed: {random_seed}, C: {C}, Kernel: {kernel}")

    # Add the accuracy to the DataFrame
    add_accuracy_to_df(selected_algo, roc_auc)

# Display the table with all algorithm accuracies
st.title("ML Algorithm Table")
if not accuracy_df.empty:
    max_roc_auc = accuracy_df["ROC AUC"].max()
    min_roc_auc = accuracy_df["ROC AUC"].min()

    # Apply conditional formatting
    def highlight_row(row):
        if row["ROC AUC"] == max_roc_auc:
            return ['color: green' if col == "ROC AUC" else '' for col in row.index]
        elif row["ROC AUC"] == min_roc_auc:
            return ['color: red' if col == "ROC AUC" else '' for col in row.index]
        else:
            return ['' for _ in row.index]

    styled_df = accuracy_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df)

# Display the hyperparameter summary table
st.title("Hyperparameter Summary Table")
if not hyperparam_df.empty:
    st.dataframe(hyperparam_df)

# Plotting Bar graph for all algorithms
st.title("ML Algorithm Bar Graph")
st.bar_chart(accuracy_df.set_index("ML Algorithm (Classification)")['ROC AUC'])

# Dropdown to select the model for download
selected_algo = st.selectbox("Select the algorithm model to download:", options=algorithms, index=0)
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