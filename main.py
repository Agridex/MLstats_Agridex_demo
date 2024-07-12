import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, Menu, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy import stats

class ModernDataAnalysisUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Agridex SStat")
        self.master.geometry("1400x800")
        self.master.configure(bg='#2C3E50')

        self.data = None
        self.feature_columns = []
        self.target_column = None
        self.model = None
        self.models_info = {}
        self.object_environment = {}

        self.create_widgets()
        self.create_menu()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', background='#3498DB', foreground='white', font=('Arial', 10, 'bold'), padding=5)
        style.configure('TLabel', background='#2C3E50', foreground='white', font=('Arial', 10))
        style.configure('TFrame', background='#2C3E50')
        style.configure('TNotebook', background='#2C3E50', foreground='white')
        style.configure('TNotebook.Tab', background='#34495E', foreground='white', padding=[10, 5])
        style.map('TNotebook.Tab', background=[('selected', '#3498DB')])

        # Main layout
        left_frame = ttk.Frame(self.master, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        right_frame = ttk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame widgets
        ttk.Button(left_frame, text="Browse Data", command=self.load_file).pack(fill=tk.X, pady=5)
        
        ttk.Label(left_frame, text="Select Features:").pack(pady=(10,0))
        self.feature_frame = ttk.Frame(left_frame)
        self.feature_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.feature_canvas = tk.Canvas(self.feature_frame, bg='#34495E')
        self.feature_scrollbar = ttk.Scrollbar(self.feature_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_scrollable_frame = ttk.Frame(self.feature_canvas)


        self.feature_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.feature_canvas.configure(
                scrollregion=self.feature_canvas.bbox("all")
            )
        )

        self.feature_canvas.create_window((0, 0), window=self.feature_scrollable_frame, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=self.feature_scrollbar.set)

        self.feature_frame.pack(fill=tk.BOTH, expand=True)
        self.feature_canvas.pack(side="left", fill=tk.BOTH, expand=True)
        self.feature_scrollbar.pack(side="right", fill="y")

        ttk.Label(left_frame, text="Select Target:").pack(pady=(10,0))
        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(left_frame, textvariable=self.target_var)
        self.target_dropdown.pack(fill=tk.X, pady=5)

        ttk.Button(left_frame, text="Convert Selected to Numeric", command=self.convert_to_numeric).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Convert Selected to Float", command=self.convert_to_float).pack(fill=tk.X, pady=5)
        ttk.Button(left_frame, text="Convert Selected to Int", command=self.convert_to_int).pack(fill=tk.X, pady=5)

        # Notebook for separating ML and Stats
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # ML Tab
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="Machine Learning")

        ttk.Label(ml_frame, text="Select ML Analysis:").pack(pady=(10,0))
        self.ml_var = tk.StringVar()
        ml_options = ["Linear Regression", "Logistic Regression", "Decision Tree Regressor", "Decision Tree Classifier",
                      "Random Forest Regressor", "Random Forest Classifier", "Gradient Boosting Regressor", 
                      "Gradient Boosting Classifier", "SVM Regressor", "SVM Classifier", 
                      "K-Nearest Neighbors Regressor", "K-Nearest Neighbors Classifier"]
        self.ml_dropdown = ttk.Combobox(ml_frame, textvariable=self.ml_var, values=ml_options)
        self.ml_dropdown.pack(fill=tk.X, pady=5)

        ttk.Button(ml_frame, text="Run ML Analysis", command=self.run_ml_analysis).pack(fill=tk.X, pady=5)

        # Stats Tab
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")

        ttk.Label(stats_frame, text="Select Statistical Analysis:").pack(pady=(10,0))
        self.stats_var = tk.StringVar()
        stats_options = ["T-Test", "ANOVA", "MANOVA", "Pivot Table", "Group Data", 
                         "Pearson Correlation", "Spearman Correlation"]
        self.stats_dropdown = ttk.Combobox(stats_frame, textvariable=self.stats_var, values=stats_options)
        self.stats_dropdown.pack(fill=tk.X, pady=5)

        ttk.Button(stats_frame, text="Run Statistical Analysis", command=self.run_statistical_analysis).pack(fill=tk.X, pady=5)

        # Plot Tab
        plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="Plots")

        ttk.Label(plot_frame, text="Select Plot:").pack(pady=(10,0))
        self.plot_var = tk.StringVar()
        plot_options = ["Correlation Heatmap", "Feature Importance", "Scatter Plot", "Box Plot", 
                        "Histogram", "Bar Chart", "Line Chart"]
        self.plot_dropdown = ttk.Combobox(plot_frame, textvariable=self.plot_var, values=plot_options)
        self.plot_dropdown.pack(fill=tk.X, pady=5)

        ttk.Button(plot_frame, text="Generate Plot", command=self.generate_plot).pack(fill=tk.X, pady=5)

        # Models and Tables in Environment
        ttk.Label(left_frame, text="Models and Tables in Environment:").pack(pady=(10,0))
        self.model_info_text = scrolledtext.ScrolledText(left_frame, height=10, width=40, bg='#34495E', fg='white')
        self.model_info_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Right frame widgets
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

        # Frame for data preview and model output
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Data preview
        data_frame = ttk.Frame(bottom_frame)
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(data_frame, text="Data Preview:").pack()
        self.data_text = scrolledtext.ScrolledText(data_frame, height=10, bg='#34495E', fg='white')
        self.data_text.pack(fill=tk.BOTH, expand=True)

        self.show_all_data_var = tk.IntVar()
        ttk.Checkbutton(data_frame, text="Show all data", variable=self.show_all_data_var, command=self.show_data_table).pack()

        # Model output
        output_frame = ttk.Frame(bottom_frame)
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(output_frame, text="Analysis Output:").pack()
        self.model_output_text = scrolledtext.ScrolledText(output_frame, height=10, bg='#34495E', fg='white')
        self.model_output_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(left_frame, text="Object Environment:").pack(pady=(10,0))
        self.object_env_text = scrolledtext.ScrolledText(left_frame, height=5, width=40, bg='#34495E', fg='white')
        self.object_env_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def create_menu(self):
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Data", command=self.load_file)
        file_menu.add_command(label="Save Data", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        data_menu = Menu(menubar, tearoff=0)
        data_menu.add_command(label="Remove Duplicates", command=self.remove_duplicates)
        data_menu.add_command(label="Fill NA Values", command=self.fill_na_values)
        data_menu.add_command(label="Normalize Data", command=self.normalize_data)
        data_menu.add_command(label="Create Dummy Variables", command=self.create_dummy_variables)
        menubar.add_cascade(label="Data", menu=data_menu)

        object_menu = Menu(menubar, tearoff=0)
        object_menu.add_command(label="Save Current Model", command=self.save_model)
        object_menu.add_command(label="Load Model", command=self.load_model)
        object_menu.add_command(label="Delete Object", command=self.delete_object)
        menubar.add_cascade(label="Object Environment", menu=object_menu)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("Text files", "*.txt"), ("CSV files", "*.csv")])
        if file_path:
            file_extension = file_path.split('.')[-1].lower()
            if file_extension == 'xlsx':
                self.data = pd.read_excel(file_path)
            elif file_extension in ['txt', 'csv']:
                self.data = pd.read_csv(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            
            self.update_feature_listbox()
            self.update_target_dropdown()
            self.show_data_table()
            self.update_model_info("Data loaded")
            self.object_environment['data'] = self.data
            self.update_object_environment()

    def update_feature_listbox(self):
        for widget in self.feature_scrollable_frame.winfo_children():
            widget.destroy()

        self.feature_vars = {}
        for column in self.data.columns:
            var = tk.BooleanVar()
            self.feature_vars[column] = var
            cb = ttk.Checkbutton(self.feature_scrollable_frame, text=column, variable=var, 
                                 command=lambda col=column: self.update_feature_columns(col))
            cb.pack(anchor="w")

    def update_feature_columns(self, column):
        if self.feature_vars[column].get():
            if column not in self.feature_columns:
                self.feature_columns.append(column)
        else:
            if column in self.feature_columns:
                self.feature_columns.remove(column)

    def update_target_dropdown(self):
        self.target_dropdown['values'] = list(self.data.columns)

    def convert_to_numeric(self):
        selected_columns = [col for col, var in self.feature_vars.items() if var.get()]
        for col in selected_columns:
            if self.data[col].dtype == 'object':
                try:
                    self.data[col] = pd.to_numeric(self.data[col], errors='raise')
                    self.model_output_text.insert(tk.END, f"Converted {col} to numeric\n")
                except ValueError:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
                    self.model_output_text.insert(tk.END, f"Converted {col} to numeric using label encoding\n")
            elif self.data[col].dtype == 'bool':
                self.data[col] = self.data[col].astype(int)
                self.model_output_text.insert(tk.END, f"Converted boolean column {col} to 0 and 1\n")
            else:
                self.model_output_text.insert(tk.END, f"Column {col} is already numeric\n")
        
        self.show_data_table()
        self.update_model_info("Data types converted")

    def show_data_table(self):
        self.data_text.delete('1.0', tk.END)
        if self.show_all_data_var.get():
            self.data_text.insert(tk.END, self.data.to_string())
        else:
            self.data_text.insert(tk.END, self.data.head(100).to_string())

    def run_ml_analysis(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        self.feature_columns = [col for col, var in self.feature_vars.items() if var.get()]
        self.target_column = self.target_var.get()

        if not self.feature_columns:
            messagebox.showerror("Error", "Please select at least one feature column")
            return

        if not self.target_column:
            messagebox.showerror("Error", "Please select a target column")
            return

        X = self.data[self.feature_columns]
        y = self.data[self.target_column]

        # Check if the target variable is categorical or continuous
        is_classification = y.dtype == 'object' or (y.dtype in ['int64', 'float64'] and len(y.unique()) <= 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_type = self.ml_var.get()
        if model_type in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", 
                        "Gradient Boosting Regressor", "SVM Regressor", "K-Nearest Neighbors Regressor"]:
            if is_classification:
                messagebox.showerror("Error", "Selected regression model but target variable appears to be categorical")
                return
                
            self.run_regression(X_train, X_test, y_train, y_test, model_type)
        elif model_type in ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier", 
                            "Gradient Boosting Classifier", "SVM Classifier", "K-Nearest Neighbors Classifier"]:
            if not is_classification:
                messagebox.showerror("Error", "Selected classification model but target variable appears to be continuous")
                return
            self.run_classification(X_train, X_test, y_train, y_test, model_type)

    def run_regression(self, X_train, X_test, y_train, y_test, model_type):
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_type == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "SVM Regressor":
            model = SVR(kernel='rbf')
        elif model_type == "K-Nearest Neighbors Regressor":
            model = KNeighborsRegressor(n_neighbors=5)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.model = model
        self.models_info[model_type] = f"MSE: {mse:.2f}, R2: {r2:.2f}"
        self.update_model_info(f"{model_type} trained")

        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, f"{model_type} Results:\nMean Squared Error: {mse:.2f}\nR-squared: {r2:.2f}")

    def run_classification(self, X_train, X_test, y_train, y_test, model_type):
        if y_train.dtype == 'object':
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
            target_names = le.classes_
        else:
            target_names = [str(c) for c in sorted(y_train.unique())]

        if model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier(random_state=42)
        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "Gradient Boosting Classifier":
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == "SVM Classifier":
            model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == "K-Nearest Neighbors Classifier":
            model = KNeighborsClassifier(n_neighbors=5)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names)
        self.model = model
        self.models_info[model_type] = f"Accuracy: {accuracy:.2f}"
        self.update_model_info(f"{model_type} trained")

        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, f"{model_type} Results:\nAccuracy: {accuracy:.2f}\n\nClassification Report:\n{report}")


    def convert_to_float(self):
        selected_columns = [col for col, var in self.feature_vars.items() if var.get()]
        for col in selected_columns:
            try:
                self.data[col] = self.data[col].astype(float)
                self.model_output_text.insert(tk.END, f"Converted {col} to float\n")
            except ValueError:
                self.model_output_text.insert(tk.END, f"Could not convert {col} to float\n")
        self.show_data_table()
        self.update_model_info("Data types converted to float")

    def convert_to_int(self):
        selected_columns = [col for col, var in self.feature_vars.items() if var.get()]
        for col in selected_columns:
            try:
                self.data[col] = self.data[col].astype(int)
                self.model_output_text.insert(tk.END, f"Converted {col} to int\n")
            except ValueError:
                self.model_output_text.insert(tk.END, f"Could not convert {col} to int\n")
        self.show_data_table()
        self.update_model_info("Data types converted to int")


    def run_statistical_analysis(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        self.feature_columns = [col for col, var in self.feature_vars.items() if var.get()]
        self.target_column = self.target_var.get()

        analysis_type = self.stats_var.get()
        if analysis_type == "T-Test":
            self.run_t_test()
        elif analysis_type == "ANOVA":
            self.run_anova()
        elif analysis_type == "MANOVA":
            self.run_manova()
        elif analysis_type == "Pivot Table":
            self.create_pivot_table()
        elif analysis_type == "Group Data":
            self.group_data()
        elif analysis_type == "Pearson Correlation":
            self.pearson_correlation()
        elif analysis_type == "Spearman Correlation":
            self.spearman_correlation()

    def run_t_test(self):
        if not self.target_column or len(self.feature_columns) != 1:
            messagebox.showerror("Error", "Please select one feature column and one target column for T-Test")
            return

        feature = self.data[self.feature_columns[0]]
        target = self.data[self.target_column]
    
        group1 = feature[target == target.unique()[0]]
        group2 = feature[target == target.unique()[1]]
    
        t_stat, p_value = stats.ttest_ind(group1, group2)
    
        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, f"T-Test Results:\nt-statistic: {t_stat:.4f}\np-value: {p_value:.4f}")

    def run_anova(self):
        if not self.target_column or len(self.feature_columns) < 1:
            messagebox.showerror("Error", "Please select at least one feature column and one target column for ANOVA")
            return
    
        try:
            formula = f"{self.target_column} ~ {' + '.join(self.feature_columns)}"
            model = ols(formula, data=self.data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
        
            self.model_output_text.delete('1.0', tk.END)
            self.model_output_text.insert(tk.END, "ANOVA Results:\n")
            self.model_output_text.insert(tk.END, anova_table.to_string())
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during ANOVA: {str(e)}")

    def run_manova(self):
        if not self.target_column or len(self.feature_columns) < 2:
            messagebox.showerror("Error", "Please select at least two feature columns and one target column for MANOVA")
            return
    
        try:
            maov = MANOVA.from_formula(f"{' + '.join(self.feature_columns)} ~ {self.target_column}", data=self.data)
            test_results = maov.mv_test()
        
            self.model_output_text.delete('1.0', tk.END)
            self.model_output_text.insert(tk.END, "MANOVA Results:\n")
            self.model_output_text.insert(tk.END, str(test_results))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during MANOVA: {str(e)}")

    def create_pivot_table(self):
        if len(self.feature_columns) < 1:
            messagebox.showerror("Error", "Please select at least one column for pivot table")
            return
    
        index = self.feature_columns[0]
    
        if len(self.feature_columns) > 1:
            columns = self.feature_columns[1]
        else:
            columns = None
    
        if self.target_column:
            values = self.target_column
        elif len(self.feature_columns) > 2:
            values = self.feature_columns[2]
        else:
            values = self.feature_columns[0]
    
        try:
            pivot = pd.pivot_table(self.data, values=values, index=index, columns=columns, aggfunc=np.mean)
        
            self.model_output_text.delete('1.0', tk.END)
            self.model_output_text.insert(tk.END, "Pivot Table:\n")
            self.model_output_text.insert(tk.END, pivot.to_string())
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while creating pivot table: {str(e)}")

    def group_data(self):
        if len(self.feature_columns) < 1:
            messagebox.showerror("Error", "Please select at least one column for grouping")
            return
    
        group_by = self.feature_columns[0]
    
        if self.target_column:
            agg_column = self.target_column
        elif len(self.feature_columns) > 1:
            agg_column = self.feature_columns[1]
        else:
            messagebox.showerror("Error", "Please select a target column or a second feature column for aggregation")
            return
    
        try:
            grouped = self.data.groupby(group_by)[agg_column].agg(['mean', 'sum', 'count'])
        
            self.model_output_text.delete('1.0', tk.END)
            self.model_output_text.insert(tk.END, f"Grouped Data (by {group_by}):\n")
            self.model_output_text.insert(tk.END, grouped.to_string())
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while grouping data: {str(e)}")

    def pearson_correlation(self):
        if not self.feature_columns:
            messagebox.showerror("Error", "Please select columns for correlation")
            return
    
        numeric_data = self.data[self.feature_columns].select_dtypes(include=[float, int])
        if numeric_data.empty:
            messagebox.showerror("Error", "No numeric columns selected")
            return
    
        corr = numeric_data.corr(method='pearson')
    
        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, "Pearson Correlation:\n")
        self.model_output_text.insert(tk.END, corr.to_string())

    def spearman_correlation(self):
        if not self.feature_columns:
            messagebox.showerror("Error", "Please select columns for correlation")
            return
    
        numeric_data = self.data[self.feature_columns].select_dtypes(include=[float, int])
        if numeric_data.empty:
            messagebox.showerror("Error", "No numeric columns selected")
            return
    
        corr = numeric_data.corr(method='spearman')
    
        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, "Spearman Correlation:\n")
        self.model_output_text.insert(tk.END, corr.to_string())

    def generate_plot(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        self.feature_columns = [col for col, var in self.feature_vars.items() if var.get()]
        self.target_column = self.target_var.get()

        plot_type = self.plot_var.get()
        
        self.ax.clear()

        if plot_type == "Correlation Heatmap":
            numeric_data = self.data[self.feature_columns].select_dtypes(include=[float, int])
            sns.heatmap(numeric_data.corr(), annot=True, ax=self.ax, cmap='coolwarm')
            self.ax.set_title('Correlation Heatmap of Numeric Columns')

        elif plot_type == "Feature Importance":
            if hasattr(self.model, 'feature_importances_'):
                importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
                importances.sort_values(ascending=True).plot(kind='barh', ax=self.ax)
                self.ax.set_title('Feature Importances')
            else:
                correlations = self.data[self.feature_columns].corrwith(self.data[self.target_column]).abs().sort_values(ascending=True)
                correlations.plot(kind='barh', ax=self.ax)
                self.ax.set_title('Feature Correlations with Target')

        elif plot_type == "Scatter Plot":
            if len(self.feature_columns) >= 2:
                if self.data[self.target_column].dtype in ['int64', 'float64']:
                    sns.scatterplot(data=self.data, x=self.feature_columns[0], y=self.feature_columns[1], 
                                    hue=self.target_column, palette='viridis', ax=self.ax)
                else:
                    sns.scatterplot(data=self.data, x=self.feature_columns[0], y=self.feature_columns[1], 
                                    hue=self.target_column, ax=self.ax)
                self.ax.set_title(f'Scatter Plot: {self.feature_columns[0]} vs {self.feature_columns[1]}')

        elif plot_type == "Box Plot":
            sns.boxplot(data=self.data, x=self.target_column, y=self.feature_columns[0], ax=self.ax)
            self.ax.set_title(f'Box Plot: {self.feature_columns[0]} by {self.target_column}')

        elif plot_type == "Histogram":
            self.data[self.feature_columns[0]].hist(ax=self.ax)
            self.ax.set_title(f'Histogram of {self.feature_columns[0]}')

        elif plot_type == "Bar Chart":
            self.data[self.feature_columns[:2]].plot(kind='bar', ax=self.ax)
            self.ax.set_title(f'Bar Chart: {self.feature_columns[0]} vs {self.feature_columns[1]}')

        elif plot_type == "Line Chart":
            self.data[self.feature_columns[:2]].plot(kind='line', ax=self.ax)
            self.ax.set_title(f'Line Chart: {self.feature_columns[0]} vs {self.feature_columns[1]}')

        self.canvas.draw()

    def update_model_info(self, message):
        self.model_info_text.delete('1.0', tk.END)
        self.model_info_text.insert(tk.END, f"Latest action: {message}\n\n")
        self.model_info_text.insert(tk.END, "Models in environment:\n")
        for model, info in self.models_info.items():
            self.model_info_text.insert(tk.END, f"- {model}: {info}\n")
        if self.data is not None:
            self.model_info_text.insert(tk.END, f"\nData shape: {self.data.shape}")
        else:
            self.model_info_text.insert(tk.END, "\nNo data loaded")

    def save_data(self):
        if self.data is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if file_path:
                self.data.to_excel(file_path, index=False)
                messagebox.showinfo("Save Successful", "Data saved successfully!")
        else:
            messagebox.showerror("Error", "No data to save")

    def save_model(self):
        if self.model is not None:
            model_name = simpledialog.askstring("Save Model", "Enter a name for the model:")
            if model_name:
                self.object_environment[model_name] = self.model
                self.update_object_environment()
                messagebox.showinfo("Success", f"Model '{model_name}' saved to object environment")
        else:
            messagebox.showerror("Error", "No model to save")

    def load_model(self):
        if self.object_environment:
            model_name = simpledialog.askstring("Load Model", "Enter the name of the model to load:")
            if model_name in self.object_environment:
                self.model = self.object_environment[model_name]
                messagebox.showinfo("Success", f"Model '{model_name}' loaded")
            else:
                messagebox.showerror("Error", f"Model '{model_name}' not found in object environment")
        else:
            messagebox.showerror("Error", "No models in object environment")
        
    def delete_object(self):
        if self.object_environment:
            object_name = simpledialog.askstring("Delete Object", "Enter the name of the object to delete:")
            if object_name in self.object_environment:
                del self.object_environment[object_name]
                self.update_object_environment()
                messagebox.showinfo("Success", f"Object '{object_name}' deleted from environment")
            else:
                messagebox.showerror("Error", f"Object '{object_name}' not found in environment")
        else:
            messagebox.showerror("Error", "No objects in environment")

    def update_object_environment(self):
        self.object_env_text.delete('1.0', tk.END)
        for obj_name, obj in self.object_environment.items():
            self.object_env_text.insert(tk.END, f"{obj_name}: {type(obj).__name__}\n")

    def remove_duplicates(self):
        if self.data is not None:
            initial_rows = len(self.data)
            self.data.drop_duplicates(inplace=True)
            final_rows = len(self.data)
            self.show_data_table()
            self.update_model_info(f"Removed {initial_rows - final_rows} duplicate rows")
        else:
            messagebox.showerror("Error", "No data loaded")

    def fill_na_values(self):
        if self.data is not None:
            for column in self.data.columns:
                if self.data[column].dtype in ['int64', 'float64']:
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                else:
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            self.show_data_table()
            self.update_model_info("Filled NA values")
        else:
            messagebox.showerror("Error", "No data loaded")

    def normalize_data(self):
        if self.data is not None:
            numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
            scaler = StandardScaler()
            self.data[numeric_columns] = scaler.fit_transform(self.data[numeric_columns])
            self.show_data_table()
            self.update_model_info("Data normalized")
        else:
            messagebox.showerror("Error", "No data loaded")

    def create_dummy_variables(self):
        if self.data is not None:
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            self.data = pd.get_dummies(self.data, columns=categorical_columns)
            self.update_feature_listbox()
            self.show_data_table()
            self.update_model_info("Created dummy variables")
        else:
            messagebox.showerror("Error", "No data loaded")

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernDataAnalysisUI(root)
    root.mainloop()
