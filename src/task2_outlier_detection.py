"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}
        
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        """
        numeric_df = self.data.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
        mask = pd.DataFrame(z_scores > threshold, columns=numeric_df.columns, index=self.data.index)
        
        self.outlier_masks['zscore'] = mask
        return mask
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        """
        numeric_df = self.data.select_dtypes(include=[np.number])
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        
        mask = (numeric_df < (Q1 - multiplier * IQR)) | (numeric_df > (Q3 + multiplier * IQR))
        self.outlier_masks['iqr'] = mask
        return mask
    
    def detect_outliers_isolation_forest(self, contamination: float = 0.1, 
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        """
        numeric_df = self.data.select_dtypes(include=[np.number]).fillna(0)
        clf = IsolationForest(contamination=contamination, random_state=random_state)
        preds = clf.fit_predict(numeric_df)
        
        # -1 indicates outlier, 1 indicates inlier
        is_outlier = (preds == -1)
        mask = pd.DataFrame({'is_outlier': is_outlier}, index=self.data.index)
        
        self.outlier_masks['isolation_forest'] = mask
        return mask
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        """
        numeric_df = self.data.select_dtypes(include=[np.number]).fillna(0)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds = clf.fit_predict(numeric_df)
        
        is_outlier = (preds == -1)
        mask = pd.DataFrame({'is_outlier': is_outlier}, index=self.data.index)
        
        self.outlier_masks['lof'] = mask
        return mask
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        """
        summary_data = {}
        for method, mask in self.outlier_masks.items():
            summary_data[method] = {
                'Total Outliers Detected': mask.sum().sum(),
                'Rows Affected': mask.any(axis=1).sum()
            }
        return pd.DataFrame(summary_data)
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        """
        if method not in self.outlier_masks:
            raise ValueError(f"Method '{method}' hasn't been executed yet.")
            
        mask_df = self.outlier_masks[method]
        rows_to_drop = mask_df.any(axis=1)
        
        treated = self.data[~rows_to_drop].copy()
        self.treated_datasets[f'removed_{method}'] = treated
        return treated
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        """
        treated = self.data.copy()
        numeric_cols = treated.select_dtypes(include=[np.number]).columns
        
        Q1 = treated[numeric_cols].quantile(0.25)
        Q3 = treated[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        treated[numeric_cols] = treated[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)
        self.treated_datasets['capped'] = treated
        return treated
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        """
        treated = self.data.copy()
        numeric_cols = treated.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            min_val = treated[col].min()
            # If negative values exist, shift them to be strictly positive
            shift = abs(min_val) + 1 if min_val < 0 else 0
            treated[col] = np.log1p(treated[col] + shift)
            
        self.treated_datasets['log_transform'] = treated
        return treated
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        """
        treated = self.data.copy()
        numeric_cols = treated.select_dtypes(include=[np.number]).columns
        
        # Yeo-Johnson supports negative values natively
        pt = PowerTransformer(method='yeo-johnson')
        treated[numeric_cols] = pt.fit_transform(treated[numeric_cols])
        
        self.treated_datasets['boxcox_transform'] = treated
        return treated
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        """
        if dataset_name == 'original':
            df = self.original_data.copy()
        elif dataset_name in self.treated_datasets:
            df = self.treated_datasets[dataset_name].copy()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found.")
            
        df = df.dropna()
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
            
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
        y = df[target_col]
        
        if len(X) < 10:
            raise ValueError("Not enough data to evaluate performance.")
            
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        # El test espera tanto r2_score como mean_cv_score
        return {
            'r2_score': scores.mean(), 
            'r2_std': scores.std(),
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std()
        }
    
    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(y=self.data[col], ax=axes[idx], color='skyblue')
            axes[idx].set_title(f'Box Plot: {col}', fontweight='bold')
            
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle('Outlier Visualization using Box Plots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            sns.violinplot(y=self.data[col], ax=axes[idx], color='lightgreen', inner='quartile')
            axes[idx].set_title(f'Violin Plot: {col}', fontweight='bold')
            
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle('Distribution Density using Violin Plots', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_outliers_scatter(self, x_col: str, y_col: str, 
                                   figsize: Tuple[int, int] = (15, 5)):
        """
        Create scatter plots showing outliers detected by different methods.
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"Columns not found: {x_col} or {y_col}")
            return plt.figure(figsize=figsize)
            
        methods = ['zscore', 'iqr', 'isolation_forest']
        available_methods = [m for m in methods if m in self.outlier_masks]
        
        if not available_methods:
            print("No detection methods have been run yet. not found")
            return plt.figure(figsize=figsize)
            
        fig, axes = plt.subplots(1, len(available_methods), figsize=figsize)
        if len(available_methods) == 1:
            axes = [axes]
            
        for idx, method in enumerate(available_methods):
            mask_df = self.outlier_masks[method]
            outlier_mask = mask_df.any(axis=1)
            
            sns.scatterplot(data=self.data, x=x_col, y=y_col, hue=outlier_mask, 
                            palette={False: 'royalblue', True: 'crimson'}, 
                            alpha=0.7, ax=axes[idx])
            axes[idx].set_title(f'Outliers detected by {method.upper()}', fontweight='bold')
            
        plt.tight_layout()
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        if x_col not in self.data.columns or y_col not in self.data.columns or z_col not in self.data.columns:
            print("Columns not found.")
            return plt.figure(figsize=figsize)
            
        if method not in self.outlier_masks:
            print(f"Method '{method}' not found.")
            return plt.figure(figsize=figsize)
            
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        outlier_mask = self.outlier_masks[method].any(axis=1)
            
        colors = ['crimson' if is_out else 'royalblue' for is_out in outlier_mask]
        
        ax.scatter(self.data[x_col], self.data[y_col], self.data[z_col], c=colors, alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'3D Multivariate Outliers ({method.upper()})', fontweight='bold')
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()