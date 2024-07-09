import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy import stats
import statsmodels.api as sm
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import colorchooser
from statsmodels.multivariate.manova import MANOVA
from tkinter import ttk, filedialog, messagebox, colorchooser, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy import stats


class DataAnalysisUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Agridex pilot")
        self.master.geometry("1400x800")
        self.master.configure(bg='#f0f0f0')

        self.data = None
        self.feature_columns = []
        self.target_column = None
        self.model = None
        self.models_info = {}

        self.create_widgets()

    def create_widgets(self):
        # Create left and right frames
        left_frame = tk.Frame(self.master, width=400, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        right_frame = tk.Frame(self.master, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame widgets
        tk.Button(left_frame, text="Browse Excel File", command=self.load_file, bg='#4CAF50', fg='white').pack(fill=tk.X, pady=5)
        
        tk.Label(left_frame, text="Select Features:", bg='#f0f0f0').pack(pady=(10,0))
        self.feature_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, height=5)
        self.feature_listbox.pack(fill=tk.X, pady=5)

        tk.Label(left_frame, text="Select Target:", bg='#f0f0f0').pack(pady=(10,0))
        self.target_var = tk.StringVar()
        self.target_dropdown = ttk.Combobox(left_frame, textvariable=self.target_var)
        self.target_dropdown.pack(fill=tk.X, pady=5)
        self.saved_features = []
        self.saved_target = None
        
        tk.Button(left_frame, text="Convert Selected to Numeric", command=self.convert_to_numeric, bg='#2196F3', fg='white').pack(fill=tk.X, pady=5)

        tk.Label(left_frame, text="Select Analysis:", bg='#f0f0f0').pack(pady=(10,0))
        self.analysis_var = tk.StringVar()
        analysis_options = ["Random Forest Regressor", "Gradient Boosting Regressor", "SVM Regressor",
                        "Random Forest Classifier", "SVM Classifier", "T-Test", "ANOVA", "MANOVA",
                        "Pivot Table", "Group Data", "Pearson Correlation", "Spearman Correlation"]
        self.analysis_dropdown = ttk.Combobox(left_frame, textvariable=self.analysis_var, values=analysis_options)
        self.analysis_dropdown.pack(fill=tk.X, pady=5)

        tk.Label(left_frame, text="Select Plot:", bg='#f0f0f0').pack(pady=(10,0))
        self.plot_var = tk.StringVar()
        plot_options = ["Correlation Heatmap", "Feature Importance", "Scatter Plot", "Box Plot", 
                    "Histogram", "Bar Chart", "Line Chart"]
        self.plot_dropdown = ttk.Combobox(left_frame, textvariable=self.plot_var, values=plot_options)
        self.plot_dropdown.pack(fill=tk.X, pady=5)

        tk.Button(left_frame, text="Run Analysis", command=self.run_analysis, bg='#FF9800', fg='white').pack(fill=tk.X, pady=5)

        # Models and Tables in Environment
        tk.Label(left_frame, text="Models and Tables in Environment:", bg='#f0f0f0').pack(pady=(10,0))
        self.model_info_text = scrolledtext.ScrolledText(left_frame, height=10, width=40)
        self.model_info_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Right frame widgets
        # Canvas for plots
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

        # Frame for data preview and model output
        bottom_frame = tk.Frame(right_frame, bg='#f0f0f0')
        bottom_frame.pack(fill=tk.BOTH, expand=True)

        # Data preview
        data_frame = tk.Frame(bottom_frame, bg='#f0f0f0')
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(data_frame, text="Data Preview:", bg='#f0f0f0').pack()
        data_container = tk.Frame(data_frame)
        data_container.pack(fill=tk.BOTH, expand=True)
        self.data_text = scrolledtext.ScrolledText(data_container, height=10)
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar = tk.Scrollbar(data_container, orient="horizontal", command=self.data_text.xview)
        data_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.data_text.configure(xscrollcommand=data_scrollbar.set)
        
        self.show_all_data_var = tk.IntVar()
        tk.Checkbutton(data_frame, text="Show all data", variable=self.show_all_data_var, command=self.show_data_table, bg='#f0f0f0').pack()

        # Model output
        output_frame = tk.Frame(bottom_frame, bg='#f0f0f0')
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(output_frame, text="Analysis Output:", bg='#f0f0f0').pack()
        output_container = tk.Frame(output_frame)
        output_container.pack(fill=tk.BOTH, expand=True)
        self.model_output_text = scrolledtext.ScrolledText(output_container, height=10)
        self.model_output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scrollbar = tk.Scrollbar(output_container, orient="horizontal", command=self.model_output_text.xview)
        output_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.model_output_text.configure(xscrollcommand=output_scrollbar.set)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.data = pd.read_excel(file_path)
            self.data = self.data.dropna()
            self.update_feature_listbox()
            self.update_target_dropdown()
            self.show_data_table()
            self.update_model_info("Data loaded")
            self.restore_selections()

    def update_feature_listbox(self):
        self.save_selections()
        self.feature_listbox.delete(0, tk.END)
        for column in self.data.columns:
            self.feature_listbox.insert(tk.END, column)
        self.restore_selections()

    def update_target_dropdown(self):
        self.save_selections()
        self.target_dropdown['values'] = list(self.data.columns)
        self.restore_selections()

    def convert_to_numeric(self):
        selected_columns = [self.feature_listbox.get(idx) for idx in self.feature_listbox.curselection()]
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

    def run_analysis(self):
        self.save_selections()

        if self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return

        self.feature_columns = self.saved_features
        self.target_column = self.saved_target

        if not self.feature_columns:
            messagebox.showerror("Error", "Please select at least one feature column")
            return

        if not self.target_column:
            messagebox.showerror("Error", "Please select a target column")
            return

        analysis_type = self.analysis_var.get()

        if analysis_type in ["Random Forest Regressor", "Gradient Boosting Regressor", "SVM Regressor"]:
            if not self.target_column:
                messagebox.showerror("Error", "Please select a target column for regression")
                return
            X = self.data[self.feature_columns]
            y = self.data[self.target_column]
            self.run_regression(X, y, analysis_type)
        elif analysis_type in ["Random Forest Classifier", "SVM Classifier"]:
            if not self.target_column:
                messagebox.showerror("Error", "Please select a target column for classification")
                return
            X = self.data[self.feature_columns]
            y = self.data[self.target_column]
            self.run_classification(X, y, analysis_type)
        elif analysis_type == "T-Test":
            self.run_t_test()
        elif analysis_type == "ANOVA":
            if not self.target_column:
                messagebox.showerror("Error", "Please select a target column for ANOVA")
                return
            self.run_anova()
        elif analysis_type == "MANOVA":
            if not self.target_column:
                messagebox.showerror("Error", "Please select a target column for MANOVA")
                return
            self.run_manova()
        elif analysis_type == "Pivot Table":
            self.create_pivot_table()
        elif analysis_type == "Group Data":
            self.group_data()
        elif analysis_type == "Pearson Correlation":
            self.pearson_correlation()
        elif analysis_type == "Spearman Correlation":
            self.spearman_correlation()

        self.plot_selected_graph()
        self.restore_selections()

    def run_regression(self, X, y, model_type):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "Gradient Boosting Regressor":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "SVM Regressor":
            model = SVR(kernel='rbf')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.model = model
        self.models_info[model_type] = f"MSE: {mse:.2f}, R2: {r2:.2f}"
        self.update_model_info(f"{model_type} trained")

        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, f"{model_type} Results:\nMean Squared Error: {mse:.2f}\nR-squared: {r2:.2f}")

    def update_feature_listbox(self):
        self.feature_listbox.delete(0, tk.END)
        for column in self.data.columns:
            self.feature_listbox.insert(tk.END, column)
    
    # Reselect previously selected features
        for i, column in enumerate(self.data.columns):
            if column in self.feature_columns:
                self.feature_listbox.selection_set(i)

    def update_target_dropdown(self):
        self.target_dropdown['values'] = list(self.data.columns)
    
    # Reselect previously selected target
        if self.target_column in self.data.columns:
            self.target_var.set(self.target_column)

    def run_classification(self, X, y, model_type):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "SVM Classifier":
            model = SVC(kernel='rbf', probability=True)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        self.model = model
        self.models_info[model_type] = f"Accuracy: {accuracy:.2f}"
        self.update_model_info(f"{model_type} trained")

        self.model_output_text.delete('1.0', tk.END)
        self.model_output_text.insert(tk.END, f"{model_type} Results:\nAccuracy: {accuracy:.2f}\n\nClassification Report:\n{report}")

    def run_t_test(self):
        if not self.target_column or len(self.feature_columns) != 1:
            messagebox.showerror("Error", "Please select one feature column and one target column for T-Test")
            return

        feature = self.data[self.feature_columns[0]]
        target = self.data[self.target_column]
    
        # Assuming binary classification in target
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
            from statsmodels.multivariate.manova import MANOVA
        
            maov = MANOVA.from_formula(f"{' + '.join(self.feature_columns)} ~ {self.target_column}", data=self.data)
            test_results = maov.mv_test()
        
            self.model_output_text.delete('1.0', tk.END)
            self.model_output_text.insert(tk.END, "MANOVA Results:\n")
            self.model_output_text.insert(tk.END, str(test_results))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during MANOVA: {str(e)}")

    def plot_selected_graph(self):
        self.ax.clear()
        color = self.pick_color()

        if self.plot_var.get() == "Correlation Heatmap":
            numeric_data = self.data[self.feature_columns].select_dtypes(include=[float, int])
            sns.heatmap(numeric_data.corr(), annot=True, ax=self.ax, cmap=plt.cm.get_cmap('coolwarm'))
            self.ax.set_title('Correlation Heatmap of Numeric Columns')

        elif self.plot_var.get() == "Feature Importance" and hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
            importances.sort_values(ascending=True).plot(kind='barh', ax=self.ax, color=color)
            self.ax.set_title('Feature Importances')

        elif self.plot_var.get() == "Scatter Plot":
            if len(self.feature_columns) >= 2:
                sns.scatterplot(data=self.data, x=self.feature_columns[0], y=self.feature_columns[1], hue=self.target_column, ax=self.ax)
                self.ax.set_title(f'Scatter Plot: {self.feature_columns[0]} vs {self.feature_columns[1]}')

        elif self.plot_var.get() == "Box Plot":
            sns.boxplot(data=self.data, x=self.target_column, y=self.feature_columns[0], ax=self.ax, color=color)
            self.ax.set_title(f'Box Plot: {self.feature_columns[0]} by {self.target_column}')

        elif self.plot_var.get() == "Histogram":
            self.plot_histogram()

        elif self.plot_var.get() == "Bar Chart":
            self.plot_bar_chart()

        elif self.plot_var.get() == "Line Chart":
            self.plot_line_chart()

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
     
    def plot_histogram(self):
         if self.data is None or len(self.feature_columns) == 0:
            messagebox.showerror("Error", "Please load data and select at least one column")
            return
    
         self.ax.clear()
         self.data[self.feature_columns[0]].hist(ax=self.ax)
         self.ax.set_title(f'Histogram of {self.feature_columns[0]}')
         self.canvas.draw()
    
    def plot_bar_chart(self):
        if self.data is None or len(self.feature_columns) < 2:
            messagebox.showerror("Error", "Please load data and select at least two columns")
            return
    
        self.ax.clear()
        self.data[self.feature_columns[:2]].plot(kind='bar', ax=self.ax)
        self.ax.set_title(f'Bar Chart: {self.feature_columns[0]} vs {self.feature_columns[1]}')
        self.canvas.draw()

    def plot_line_chart(self):
        if self.data is None or len(self.feature_columns) < 2:
            messagebox.showerror("Error", "Please load data and select at least two columns")
            return
    
        self.ax.clear()
        self.data[self.feature_columns[:2]].plot(kind='line', ax=self.ax)
        self.ax.set_title(f'Line Chart: {self.feature_columns[0]} vs {self.feature_columns[1]}')
        self.canvas.draw()

    def pick_color(self):
        color = colorchooser.askcolor(title="Choose color")[1]
        return color if color else 'blue'  

    def save_selections(self):
        self.saved_features = [self.feature_listbox.get(idx) for idx in self.feature_listbox.curselection()]
        self.saved_target = self.target_var.get()

    def restore_selections(self):
        if hasattr(self, 'saved_features'):
            for i, feature in enumerate(self.feature_listbox.get(0, tk.END)):
                if feature in self.saved_features:
                    self.feature_listbox.selection_set(i)
        if hasattr(self, 'saved_target'):
            self.target_var.set(self.saved_target)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataAnalysisUI(root)
    root.mainloop()
