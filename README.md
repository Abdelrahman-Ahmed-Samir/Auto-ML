# Auto-ML

This Python GUI application provides a user-friendly interface for performing several key tasks related to working with CSV data files, including preprocessing and applying machine learning algorithms. The primary features of this application are as follows:
1.	Data Loading and Display: Users can select a CSV file from their local system using the "Select CSV File" button. The selected file is loaded into the application, and its contents are displayed within the GUI.
2.	Data Preprocessing: The application offers various data preprocessing options, which users can select via checkboxes. These options include handling missing values, removing duplicate rows, scaling numeric features, and one-hot encoding categorical columns. Users can customize which preprocessing steps to apply to their data.
3.	Machine Learning Algorithms: Users can choose from a list of machine learning algorithms to apply to their data. Supported algorithms include Decision Trees, Linear Regression, K-Nearest Neighbors, Naive Bayes (for classification tasks), and K-Means clustering.
4.	Algorithm-Specific Output: Depending on the selected machine learning algorithm, the application provides algorithm-specific output metrics. For example, it calculates and displays accuracy, classification reports, mean squared error, and R^2 score. For Decision Trees, it also generates and visualizes a confusion matrix as a heatmap.
5.	Image Classification Support: If a user selects an image file (e.g., JPG or PNG), the application provides a mechanism for image classification. Users can load an image file and apply a pre-trained image-based model for classification.
6.	Text Data Preprocessing: For text-based data, the application includes text preprocessing functionality. It converts text to lowercase, tokenizes it into words, removes stopwords and non-alphabetic tokens, and applies stemming to reduce words to their root form.
7.	Customization and Visualization: The application allows users to customize preprocessing and algorithm selection based on their data's characteristics. It also visualizes key results, such as confusion matrices for classification tasks.
8.	User-Friendly Interface: The application is built using the Tkinter library, providing a user-friendly and interactive graphical interface for users to perform data-related tasks.



# Dataset Description:
Used datasets to test the program and how it performs are:

1)	Email Spam Detection Dataset: Containing related information of 5172 randomly picked email files and their respective labels for spam or not-spam classification.
2)	Document Classification Dataset: There are 3 main classes, slides, thesis and research. Slides corresponds to slides documents from SlideShare, thesis for student thesis paper in CORE and research for any other article that is neither a set of slides or a student thesis paper.
3)	Platform Price Prediction Dataset: I used salary and bikes prices datasets


Link of Dataset:
1) Email Spam Detection Dataset (classification) | Kaggle
2) Document classification from aggregated features | Kaggle
3) Salary_Data | Kaggle , (Can’t find the motorcycle dataset it was an old dataset I found in my pc)
