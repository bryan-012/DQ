import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import re
class DataQualityApp:
    def __init__(self, master):
        self.master = master
        master.title("Data Quality Visualization App")
        
        self.master.geometry("800x800")  # Width and height
        # Initialize DataFrame
        self.df = None

        self.import_button = tk.Button(master, text="Import CSV", command=self.import_csv)
        self.import_button.pack(pady=10)
        
        # Overall Data Quality button
        self.quality_button = tk.Button(master, text="Overall Data Quality", command=self.plot_quality)
        self.quality_button.pack(pady=10)
        

       # Variables dropdown menu
        self.var_label = tk.Label(master, text="Select Variable:")
        self.var_label.pack(pady=10)
        self.selected_var = tk.StringVar()
        self.var_dropdown = tk.OptionMenu(master, self.selected_var, [])
        self.var_dropdown.pack()
        
       
        
        # Variable Graph button
        self.var_button = tk.Button(master, text="Variable Graph", command=self.plot_variable2)
        self.var_button.pack(pady=10)
        
        # Plot window
        self.plot_window = None
    def update_var_dropdown(self):
        if self.df is None:
            return
        self.var_dropdown["menu"].delete(0, "end")
        for col in self.df.columns:
            self.var_dropdown["menu"].add_command(label=col, command=tk._setit(self.selected_var, col))
        
        self.var_dropdown.bind("<Button-1>", lambda event: self.update_var_dropdown())
        
    def import_csv(self):
        filename = filedialog.askopenfilename()
        if filename.endswith('.csv'):
            self.df = pd.read_csv(filename)
            self.var_dropdown["menu"].delete(0, "end")
            for col in self.df.columns:
                self.var_dropdown["menu"].add_command(label=col, command=tk._setit(self.selected_var, col))
            print("CSV file imported successfully!")
        else:
            print("Please select a CSV file.")
    
    def plot_quality1(self):
        if self.df is None:
            print("Please import a CSV file.")
            return
        # Destroy previous plot window if it exists
        # Destroy any existing plot window
        if self.plot_window:
            self.plot_window.destroy()

           # Create new plot window with updated graph
        self.plot_window = tk.Toplevel(self.master)
        self.plot_window.title("Variable Graph")
        # Clear the plot window frame
        plot_frame = tk.Frame(self.plot_window)
        plot_frame.pack(side='right', fill='both', expand=True)
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 10))
        ax1.plot(self.df.isnull().sum())
        ax1.set_xlabel("Variables")
        ax1.set_ylabel("Missing Values")
        ax1.set_title("Overall Data Quality - Line Chart")
        ax2.bar(self.df.columns, self.df.isnull().sum())
        ax2.set_xlabel("Variables")
        ax2.set_ylabel("Missing Values")
        ax2.set_title("Overall Data Quality - Bar Chart")
        self.plot_window = tk.Toplevel(self.master)
        self.plot_window.title("Overall Data Quality Graphs")
        
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
       
        
       
        
       
     
        
    
    def plot_quality(self):
        if self.df is None:
            print("Please import a CSV file.")
            return
    
    # Destroy previous plot window if it exists
        if self.plot_window:
            self.plot_window.destroy()

    # Create new plot window with updated graph
        self.plot_window = tk.Toplevel(self.master)
        self.plot_window.title("Data Quality Graphs")

    # Create a dropdown menu for selecting the quality metric
        quality_var = tk.StringVar()
        quality_var.set("Missing Values")
        quality_options = ["Missing Values", "Row Count", "Distinct Count","Null Count","Unique Values", "Duplicate Count","Blank Count", "Pattern Frequency","Accuracy","Uniqueness"]
        quality_dropdown = tk.OptionMenu(self.plot_window, quality_var, *quality_options)
        quality_dropdown.pack(side='top', padx=10, pady=10)

    # Clear the plot window frame
        plot_frame = tk.Frame(self.plot_window)
        plot_frame.pack(side='left', fill='both', expand=True)

    # Create a function to plot the selected quality metric
        def plot_metric():
            metric = quality_var.get()
   
   # Clear the previous plot
            for widget in plot_frame.winfo_children():
                widget.destroy()
            
            fig, ax = plt.subplots(figsize=(15, 10))
            if metric == "Missing Values":
                
                
                ax.bar(self.df.columns, self.df.isnull().sum())
                ax.set_xticklabels(self.df.columns,rotation=90)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Missing Values")
                ax.set_title("Missing Values by Variable")
            elif metric == "Row Count":
                ax.bar(self.df.columns, len(self.df))
                ax.set_xticklabels(self.df.columns,rotation=90)
                #ax.barh([""], [len(self.df)])
                ax.set_xlabel("Row Count")
                ax.set_title("Number of Rows")
            elif metric == "Distinct Count":
                ax.bar(self.df.columns, self.df.nunique(dropna=False))
                ax.set_xticklabels(self.df.columns,rotation=90)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Distinct Count")
                ax.set_title("Distinct Count by Variable")
                

            elif metric == "Null Count":
                null_count = (self.df == '0').sum()
                ax.bar(null_count.index, null_count.values)
                ax.set_xticklabels(null_count.index, rotation=90)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Null Count")
                ax.set_title("Null Count by Variable")
            elif metric == "Unique Values":
                ax.bar(self.df.columns, sum(self.df.value_counts() == 1))
                ax.set_xticklabels(self.df.columns,rotation=90)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Unique Values")
                ax.set_title("Unique Values by Variable")
            elif metric == "Duplicate Count":
                
                counts = self.df.apply(lambda x: len(x) - x.nunique())
                ax.bar(counts.index, counts.values)
                
                ax.set_xticklabels(self.df.columns,rotation=90)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Duplicate Count")
                ax.set_title("Duplicate Count by Variable")
             
            elif metric == "Blank Count":
                ax.bar(range(len(self.df.columns)), self.df.isna().sum())
                ax.set_xticks(range(len(self.df.columns)))
                ax.set_xticklabels(self.df.columns,rotation=90)
                ax.set_xlabel("Variables",rotation=90)
                ax.set_ylabel("Blank Count")
                ax.set_title("Blank Count by Variable")
            elif metric == "Pattern Frequency1":
                ax.bar(self.df.columns, self.df.value_counts())
                ax.set_xlabel("Variables")
                ax.set_ylabel("Pattern Frequency")
                ax.set_title("Pattern Frequency by Variable")

            elif metric == "Pattern Frequency":
            # Count the number of occurrences of each pattern
                pattern_counts = {}
                for col in self.df.columns:
                    unique_values = self.df[col].dropna().unique()
                    patterns = {}
                    for value in unique_values:
                        pattern = re.sub(r'\d+', '#', str(value))  # Replace digits with # in the value
                        if pattern in patterns:
                            patterns[pattern] += 1
                        else:
                            patterns[pattern] = 1
                    pattern_counts[col] = patterns
            
            # Plot the pattern frequencies
                pattern_df = pd.DataFrame.from_dict(pattern_counts)
                pattern_df.plot(kind='bar', ax=ax, rot=90)
                ax.set_xlabel("Variables",rotation=90)
                ax.set_ylabel("Frequency")
                ax.set_title("Pattern Frequency by Variable")
                
           

            elif metric == "Accuracy":
                accuracy_scores = []
                for column in self.df.columns:
                    if self.df[column].dtype in ['int64', 'float64']:
                        accuracy_scores.append(1 - (self.df[column].isnull().sum() / len(self.df[column])))
                    else:
                        non_null_count = self.df[column].count()
                        unique_values = self.df[column].nunique()
                        if unique_values == 1:
                            accuracy_scores.append(1)
                        else:
                            accuracy_scores.append(non_null_count / unique_values)
                ax.bar(self.df.columns, accuracy_scores)
                ax.set_xticklabels(self.df.columns,rotation=90)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Accuracy Score")
                ax.set_title("Accuracy Score by Variable")
                
                
            elif metric == "Uniqueness":
                unique_ratios = []
                for column in self.df.columns:
                    unique_count = self.df[column].nunique()
                    total_count = len(self.df[column])
                    unique_ratio = unique_count / total_count
                    unique_ratios.append(unique_ratio)
                    ax.bar(self.df.columns, unique_ratios)
                ax.set_xlabel("Variables")
                ax.set_ylabel("Uniqueness Ratio")
                ax.set_title("Uniqueness Ratio by Variable")
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()   
             
            
               
    # Create a button to update the plot
        update_button = tk.Button(self.plot_window, text="Update Plot", command=plot_metric)
        update_button.pack(side='top', padx=10, pady=10)

    # Plot the initial metric
        plot_metric()
    
    
    
    
    
    
    def plot_variable2(self):
        if self.df is None:
            print("Please import a CSV file first.")
            return
        # Destroy previous plot window if it exists

        
        var = self.selected_var.get()
        if var not in self.df.columns:
            print("Selected variable is not present in the dataset.")
            return
    

          
        
        var = self.selected_var.get()
    
        #Calculate row count and null count
        
        row_count = len(self.df)
        null_count = len(self.df[self.df[var] == '0'])



        if self.df[var].dtype == np.object:  # Handle string data
            distinct_count = len(self.df[var].unique())
            unique_values = sum(self.df[var].value_counts() == 1)
            duplicate_count = len(self.df[var]) - unique_values
            blank_count = len(self.df[self.df[var].isnull()])
            pattern_frequency = self.df[var].value_counts()
            accuracy = (row_count - null_count) / row_count if row_count > 0 else 0
            consistency = (row_count - null_count) / row_count if row_count > 0 else 0
            uniqueness = distinct_count / row_count if row_count > 0 else 0
        else:  # Handle numeric and special characters data
            distinct_count = len(self.df[var].unique())
            unique_values = sum(self.df[var].value_counts() == 1)
            duplicate_count = len(self.df[var]) - unique_values
            blank_count = len(self.df[self.df[var].isnull()])
            pattern_frequency = self.df[var].value_counts()
            accuracy = (row_count - null_count) / row_count if row_count > 0 else 0
            consistency = (self.df[var].count() - self.df[var].nunique()) / row_count if row_count > 0 else 0
            uniqueness = distinct_count / row_count if row_count > 0 else 0
            
           
        
        # Create DataFrame for quality metrics
        df_metrics = pd.DataFrame({
            'Metrics': ['RowCount', 'NullCount', 'DistinctCount', 'UniqueCount', 'DuplicateCount', 'BlankCount', 'PatternFrequency', 'Accuracy', 'Consistency',  'Uniqueness'],
            'Values': [row_count, null_count, distinct_count, unique_values, duplicate_count, blank_count, pattern_frequency.max(),accuracy, consistency, uniqueness],
            
            '%': [f"{row_count / row_count:.2%}", f"{null_count / row_count:.2%}", f"{distinct_count / row_count:.2%}", f"{unique_values / row_count:.2%}", f"{duplicate_count / row_count:.2%}", f"{blank_count / row_count:.2%}", f"{pattern_frequency.max() / row_count:.2%}", f"{accuracy / row_count:.2%}", f"{consistency / row_count:.2%}", f"{uniqueness / row_count:.2%}"]
            
        
        })
        
        # Destroy any existing plot window
        if self.plot_window:
            self.plot_window.destroy()

           # Create new plot window with updated graph
        self.plot_window = tk.Toplevel(self.master)
        self.plot_window.title("Variable Graph")
        # Clear the plot window frame
        
        table_frame = tk.Frame(self.plot_window)
        table_frame.pack(side='left', fill='y')
        table_label = tk.Label(table_frame, text="Quality Metrics")
        table_label.pack()
        treeview = ttk.Treeview(table_frame, columns=['Values', '%'])
        treeview.heading("#0", text='Metrics')
        treeview.heading("Values", text='Values')
        treeview.heading("%", text='%')
        for _, row in df_metrics.iterrows():
            treeview.insert("", "end", text=row['Metrics'], values=(row['Values'], row['%']))
        treeview.pack()

     
        
        plot_frame = tk.Frame(self.plot_window)
        plot_frame.pack(side='right', fill='both', expand=True)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(15, 8))
        x_labels = [row['Metrics'] for _, row in df_metrics.iterrows()]
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'teal','cyan'][:len(df_metrics)]
        ax.bar(x_labels,  [row_count, null_count, distinct_count, unique_values, duplicate_count, blank_count, pattern_frequency.max(), accuracy, consistency, uniqueness], color=colors)
        
        ax.set_title(f"Data Quality Metrics for {var}")
        ax.set_ylabel("Count")
        ax.set_ylim([0, row_count])
        
        
        # Add annotations on top of the bars
        for i, v in enumerate([row_count, null_count, distinct_count, unique_values, duplicate_count, blank_count, pattern_frequency.max(),accuracy, consistency, uniqueness]):
            ax.text(i, v + 0.05 * row_count, str(v), ha='center', fontweight='bold')
        index = 0
        # Add click event handler
        def on_click(event):
            nonlocal index
            if event.button == 1:  # left click
                index = int(round(event.xdata))
                value = [row_count, null_count, distinct_count, unique_values, duplicate_count, blank_count, pattern_frequency.max(), accuracy, consistency, uniqueness][index]
                ax.text(index, value + 0.05 * row_count, str(value), ha='center', va='bottom', fontweight='bold', color='red')
                canvas.draw()
            elif event.button == 3:  # right click
                menu = tk.Menu(self.plot_window, tearoff=0)
                metric = ['RowCount', 'NullCount', 'DistinctCount', 'UniqueCount', 'DuplicateCount', 'BlankCount', 'PatternFrequency', 'Accuracy', 'Consistency',  'Uniqueness'][index]
                menu.add_command(label=f"View {metric} Values", command=lambda: self.view_values(var, metric))
                menu.add_command(label=f"View {metric} Rows", command=lambda: self.view_rows(var, metric))
                menu.post(event.x, event.y)
        

        canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        canvas.mpl_connect('button_release_event', on_click)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
    def view_values(self, var, metric):
        values = self.df[var].values
        if metric == 'RowCount':
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {len(values)} rows.")
            
        elif metric == 'NullCount':
            null_values = len(self.df[self.df[var] == '0'])
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {null_values} null values.")
            
        elif metric == 'DistinctCount':
            # Convert values to strings before sorting
            values = [str(value) for value in self.df[var].dropna().values]
            
            distinct_values = len(np.unique(values))
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {distinct_values} distinct values.")
            
        elif metric == 'UniqueValues':
            # Convert values to strings before sorting
            values = [str(value) for value in self.df[var].dropna().values]
            unique_values = len(sum(self.df[var].value_counts() == 1))
            
            
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {unique_values} unique values: {unique_values}")
        elif metric == 'DuplicateCount':
            duplicated = self.df.duplicated(subset=[var], keep=False)
            duplicate_values = values[duplicated]
            rows = self.df[self.df[var].isin(duplicate_values)]
            duplicate_count = len(duplicate_values)
            
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {duplicate_count} duplicate values.")
            
        elif metric == 'BlankCount':
            blank_count = np.sum(pd.isnull(values)) + len(values[values == ''])
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {blank_count} blank values.")
        elif metric == 'PatternFrequency':
        
            pattern_frequency = 0
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has a pattern frequency of {pattern_frequency}.")
       
        elif metric == 'Accuracy':
            accuracy = (len(values) - np.sum(pd.isnull(values))) / len(values)
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has an accuracy of {accuracy:.2f}.")
        elif metric == 'Consistency':
            consistency = len(values) == len(self.df)
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has a consistency of {consistency}.")
        elif metric == 'Uniqueness':
            uniqueness = len(values) == len(np.unique(values))
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has a uniqueness of {uniqueness}.")
        else:
            messagebox.showerror(title="Error", message=f"Metric '{metric}' not recognized.")

    def view_rows(self, var, metric):
        if metric == 'RowCount':
            rows=self.df
            
        elif metric == 'NullCount':
            rows = self.df[self.df[var] == '0']

        elif metric == 'DistinctCount':
            rows = self.df[self.df[var].notnull()].drop_duplicates(subset=[var])
            distinct_values = len(rows)
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {distinct_values} distinct values.")
            
         
            
        elif metric == 'UniqueValues':
            rows = self.df[var].value_counts() == 1
            unique_values = sum(rows)
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {unique_values} unique values.")
            
        elif metric == 'DuplicateCount':
            
            duplicated_rows = self.df[self.df.duplicated(subset=[var], keep=False)]
            messagebox.showinfo(title=f"{var} {metric}", message=f"{var} has {len(duplicated_rows)} duplicate values.")
            rows = duplicated_rows.sort_values(by=[var])
            
        elif metric == 'BlankCount':
            blank_rows = self.df[(self.df[var].isnull()) | (self.df[var] == '')]
            rows = blank_rows.sort_values(by=[var])
        elif metric == 'PatternFrequency':
            
            pattern_rows = self.df
            rows = pattern_rows.sort_values(by=[var])
        
        elif metric == 'Accuracy':
            accuracy = (len(self.df[var]) - np.sum(pd.isnull(self.df[var]))) / len(self.df[var])
            rows = self.df[self.df[var].isnull()]
            if accuracy < 1:
                rows = self.df[self.df[var].notnull()]
        elif metric == 'Consistency':
            consistency = len(self.df[var]) == len(self.df)
            rows = self.df[self.df[var].isnull()]
            if consistency:
                rows = self.df[self.df[var].notnull()]
        elif metric == 'Uniqueness':
            uniqueness = len(self.df[var]) == len(np.unique(self.df[var]))
            rows = self.df.drop_duplicates(subset=[var], keep=False)
            if uniqueness:
                rows = self.df[self.df[var].duplicated(keep=False)]
            
        else:
            messagebox.showerror(title="Error", message=f"Metric '{metric}' not recognized.")
            return
        rows_window = tk.Toplevel(self.plot_window)
        rows_window.title(f"{var} {metric} Rows")
        treeview = ttk.Treeview(rows_window, columns=list(self.df.columns), show='headings')
        for col in self.df.columns:
            treeview.heading(col, text=col)
        for _, row in rows.iterrows():
            treeview.insert("", "end", values=list(row))
        treeview.pack(fill='both', expand=True) 
root = tk.Tk()

app = DataQualityApp(root)
root.mainloop()


