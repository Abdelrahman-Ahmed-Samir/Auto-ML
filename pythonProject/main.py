import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB  # Import Naive Bayes for classification
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

ALGORITHMS = {
    "Decision Tree": DecisionTreeClassifier(),
    "Linear Regression": LinearRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": MultinomialNB(),  # Add Naive Bayes here
    "K-Means": KMeans(),
}

def select_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        label.config(text=f"Selected File: {file_path}")
        load_and_display_data(file_path)
    else:
        label.config(text="No file selected.")


def load_and_display_data(file_path):
    global df
    try:
        # Try reading the CSV file with UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 encoding fails, try 'latin1' (or 'ISO-8859-1') encoding
        df = pd.read_csv(file_path, encoding='latin1')

    display_data(df)

# Preproccessing Phases
def handle_missing_values():
    global df
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    display_data(df)


def remove_duplicates():
    global df
    df.drop_duplicates(inplace=True)
    display_data(df)

def scale_features():
    global df
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    display_data(df)

def preprocess_data():
    global df
    if missing_values_var.get():
        handle_missing_values()
    if duplicates_var.get():
        remove_duplicates()
    if scaling_var.get():
        scale_features()

    if one_hot_encoding_var.get():
        # Perform one-hot encoding for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    display_data(df)

def display_data(df):
    # Display data in the GUI (e.g., in a text widget or table)
    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, df.to_string())

selected_algorithm = None

# Create a function to apply K-Means clustering
def apply_kmeans(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_assignments = kmeans.fit_predict(data)
    return cluster_assignments

def on_algorithm_selected(event):
    global selected_algorithm
    selected_algorithm = algorithm_var.get()
    print(f"Selected Algorithm: {selected_algorithm}")

def get_selected_algorithm():
    global selected_algorithm
    return ALGORITHMS.get(selected_algorithm)


def train_and_evaluate_model():
    global df, selected_algorithm
    if selected_algorithm is None:
        return  # No algorithm selected, do nothing

    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train the selected algorithm
    model = get_selected_algorithm()
    if selected_algorithm == "Linear Regression":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate accuracy (for classification) or other relevant metrics (for regression)
    if selected_algorithm == "Decision Tree":
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results_text = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}"
        # Calculate and display the confusion matrix
        confusion = confusion_matrix(y_test, y_pred)
        results_text += "\nConfusion Matrix:\n"
        results_text += str(confusion)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    elif selected_algorithm == "Linear Regression":
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results_text = f"Mean Squared Error: {mse}\nR^2 Score: {r2}"
    elif selected_algorithm == "K-Nearest Neighbors":
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1.0)  # Set zero_division to 1.0
        results_text = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}"
        # Calculate and display the confusion matrix
        confusion = confusion_matrix(y_test, y_pred)
        results_text += "\nConfusion Matrix:\n"
        results_text += str(confusion)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    elif selected_algorithm == "Naive Bayes":
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1.0)  # Set zero_division to 1.0
        results_text = f"Accuracy: {accuracy}\n\nClassification Report:\n{report}"
        # Calculate and display the confusion matrix
        confusion = confusion_matrix(y_test, y_pred)
        results_text += "\nConfusion Matrix:\n"
        results_text += str(confusion)

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    elif selected_algorithm == "K-Means":
        num_clusters = 3  # You can change the number of clusters as needed
        cluster_assignments = apply_kmeans(X, num_clusters)
        results_text = f"Cluster Assignments: {cluster_assignments}"

    # Display results in the GUI
    text_widget.delete("1.0", tk.END)
    text_widget.insert(tk.END, f"Selected Algorithm: {selected_algorithm}\n")
    text_widget.insert(tk.END, results_text)



app = tk.Tk()
app.title("Data System")

file_path = ""
df = None

label = tk.Label(app, text="")
label.pack(pady=10)

select_button = tk.Button(app, text="Select CSV File", command=select_file)
preprocess_button = tk.Button(app, text="Preprocess Data", command=preprocess_data)
train_button = tk.Button(app, text="Train and Evaluate Model", command=train_and_evaluate_model)

select_button.pack()
preprocess_button.pack()
train_button.pack()

missing_values_var = tk.IntVar()
duplicates_var = tk.IntVar()
scaling_var = tk.IntVar()
one_hot_encoding_var = tk.IntVar()
one_hot_encoding_checkbox = tk.Checkbutton(app, text="One-Hot Encoding", variable=one_hot_encoding_var)
one_hot_encoding_checkbox.pack()

missing_values_checkbox = tk.Checkbutton(app, text="Handle Missing Values", variable=missing_values_var)
duplicates_checkbox = tk.Checkbutton(app, text="Remove Duplicates", variable=duplicates_var)
scaling_checkbox = tk.Checkbutton(app, text="Scale Features", variable=scaling_var)

missing_values_checkbox.pack()
duplicates_checkbox.pack()
scaling_checkbox.pack()


algorithm_var = tk.StringVar()
algorithm_dropdown = ttk.Combobox(app, textvariable=algorithm_var, values=list(ALGORITHMS.keys()))
algorithm_dropdown.bind("<<ComboboxSelected>>", on_algorithm_selected)
algorithm_dropdown.pack()  # Add this line to pack the Combobox




text_widget = tk.Text(app, height=50, width=100)  # Adjust the height and width as needed
text_widget.pack()
app.configure(bg='lightblue')  # Change the background color of the main window
label.configure(bg='lightblue')  # Change the background color of a label
label.configure(font=("Helvetica", 14), fg='darkblue')  # Set a custom font and text color for a label


app.mainloop()
