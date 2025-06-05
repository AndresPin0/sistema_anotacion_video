
"""
Análisis Exploratorio de Datos
==============================

Este script realiza un análisis exploratorio completo de los datos extraídos
para el sistema de anotación de video, incluyendo visualizaciones, estadísticas
y análisis de calidad de datos.

Funcionalidades:
- Análisis estadístico descriptivo
- Visualizaciones de distribuciones de datos
- Análisis de correlaciones entre características
- Detección de outliers
- Análisis de calidad y completitud de datos
- Comparación entre actividades

Uso:
    python data_analysis.py
    python data_analysis.py --data_dir data/training_data --output_dir analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

class DataAnalyzer:
    """
    Analizador de datos para el sistema de anotación de video.
    """
    
    def __init__(self, data_dir="data/training_data", output_dir="analysis"):
        """
        Inicializa el analizador de datos.
        
        Args:
            data_dir (str): Directorio con los datos a analizar
            output_dir (str): Directorio donde guardar resultados del análisis
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "statistics").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        
        self.df = None
        self.feature_columns = None
        self.activity_column = 'activity'
        
        
        self.analysis_results = {}
    
    def load_data(self, file_path=None):
        """
        Carga los datos para análisis.
        
        Args:
            file_path (str): Ruta específica al archivo de datos
        """
        print("=== Cargando datos para análisis ===")
        
        if file_path is None:
            
            possible_files = [
                self.data_dir / "complete_dataset.csv",
                self.data_dir / "processed" / "training_features.csv"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    break
            else:
                raise FileNotFoundError(f"No se encontraron datos en {self.data_dir}")
        
        print(f"Cargando datos desde: {file_path}")
        
        self.df = pd.read_csv(file_path)
        print(f"Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        
        
        exclude_columns = ['video_path', 'frame_idx', 'timestamp', self.activity_column]
        self.feature_columns = [col for col in self.df.columns if col not in exclude_columns]
        
        print(f"Columnas de características: {len(self.feature_columns)}")
        print(f"Actividades encontradas: {self.df[self.activity_column].nunique()}")
        
        
        self.analysis_results['basic_info'] = {
            'total_samples': len(self.df),
            'total_features': len(self.feature_columns),
            'activities': sorted(self.df[self.activity_column].unique()),
            'samples_per_activity': self.df[self.activity_column].value_counts().to_dict()
        }
    
    def descriptive_statistics(self):
        """
        Calcula estadísticas descriptivas del dataset.
        """
        print("\n=== Análisis estadístico descriptivo ===")
        
        
        activity_stats = {}
        
        for activity in self.df[self.activity_column].unique():
            activity_data = self.df[self.df[self.activity_column] == activity][self.feature_columns]
            
            stats_dict = {
                'count': len(activity_data),
                'mean': activity_data.mean().to_dict(),
                'std': activity_data.std().to_dict(),
                'min': activity_data.min().to_dict(),
                'max': activity_data.max().to_dict(),
                'median': activity_data.median().to_dict()
            }
            
            activity_stats[activity] = stats_dict
        
        self.analysis_results['descriptive_stats'] = activity_stats
        
        
        feature_stats = self.df[self.feature_columns].describe()
        
        
        stats_path = self.output_dir / "statistics" / "descriptive_stats.csv"
        feature_stats.to_csv(stats_path)
        print(f"Estadísticas descriptivas guardadas: {stats_path}")
        
        
        self._plot_feature_distributions()
        
        return feature_stats
    
    def _plot_feature_distributions(self, max_features=20):
        """
        Crea histogramas de las distribuciones de características.
        
        Args:
            max_features (int): Máximo número de características a mostrar
        """
        
        feature_vars = self.df[self.feature_columns].var().sort_values(ascending=False)
        top_features = feature_vars.head(max_features).index.tolist()
        
        
        n_cols = 4
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                ax = axes[i]
                
                
                for activity in self.df[self.activity_column].unique():
                    activity_data = self.df[self.df[self.activity_column] == activity][feature]
                    ax.hist(activity_data, alpha=0.6, label=activity, bins=20)
                
                ax.set_title(f'Distribución de {feature}')
                ax.set_xlabel('Valor')
                ax.set_ylabel('Frecuencia')
                ax.legend()
        
        
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / "feature_distributions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Distribuciones de características guardadas: {plot_path}")
    
    def correlation_analysis(self):
        """
        Analiza correlaciones entre características.
        """
        print("\n=== Análisis de correlaciones ===")
        
        
        correlation_matrix = self.df[self.feature_columns].corr()
        
        
        high_corr_pairs = []
        threshold = 0.8
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        print(f"Encontradas {len(high_corr_pairs)} correlaciones altas (|r| > {threshold})")
        
        
        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs)
            corr_path = self.output_dir / "statistics" / "high_correlations.csv"
            corr_df.to_csv(corr_path, index=False)
            print(f"Correlaciones altas guardadas: {corr_path}")
        
        
        self._plot_correlation_heatmap(correlation_matrix)
        
        self.analysis_results['correlation_analysis'] = {
            'high_correlation_count': len(high_corr_pairs),
            'correlation_threshold': threshold,
            'high_correlations': high_corr_pairs
        }
        
        return correlation_matrix
    
    def _plot_correlation_heatmap(self, correlation_matrix, max_features=50):
        """
        Crea un heatmap de la matriz de correlación.
        
        Args:
            correlation_matrix (DataFrame): Matriz de correlación
            max_features (int): Máximo número de características a mostrar
        """
        
        if len(correlation_matrix) > max_features:
            
            feature_vars = self.df[self.feature_columns].var().sort_values(ascending=False)
            top_features = feature_vars.head(max_features).index.tolist()
            correlation_subset = correlation_matrix.loc[top_features, top_features]
        else:
            correlation_subset = correlation_matrix
        
        plt.figure(figsize=(15, 12))
        
        
        mask = np.triu(np.ones_like(correlation_subset, dtype=bool))
        sns.heatmap(
            correlation_subset,
            mask=mask,
            annot=False,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Matriz de Correlación de Características')
        plt.tight_layout()
        
        plot_path = self.output_dir / "plots" / "correlation_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap de correlaciones guardado: {plot_path}")
    
    def outlier_analysis(self):
        """
        Detecta y analiza outliers en los datos.
        """
        print("\n=== Análisis de outliers ===")
        
        outlier_results = {}
        
        
        for feature in self.feature_columns:
            feature_data = self.df[feature].dropna()
            
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
            
            outlier_results[feature] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(feature_data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        
        outlier_df = pd.DataFrame(outlier_results).T
        outlier_df = outlier_df.sort_values('percentage', ascending=False)
        
        
        outlier_path = self.output_dir / "statistics" / "outlier_analysis.csv"
        outlier_df.to_csv(outlier_path)
        print(f"Análisis de outliers guardado: {outlier_path}")
        
        
        top_outlier_features = outlier_df.head(10)
        print("Características con más outliers:")
        print(top_outlier_features[['count', 'percentage']].to_string())
        
        
        self._plot_outliers(top_outlier_features.index.tolist()[:8])
        
        self.analysis_results['outlier_analysis'] = {
            'total_features_analyzed': len(outlier_results),
            'features_with_outliers': len(outlier_df[outlier_df['count'] > 0]),
            'average_outlier_percentage': outlier_df['percentage'].mean()
        }
    
    def _plot_outliers(self, features):
        """
        Crea boxplots para visualizar outliers.
        
        Args:
            features (list): Lista de características a visualizar
        """
        n_cols = 2
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                
                activity_data = []
                activity_labels = []
                
                for activity in self.df[self.activity_column].unique():
                    data = self.df[self.df[self.activity_column] == activity][feature].dropna()
                    activity_data.append(data)
                    activity_labels.append(activity)
                
                ax.boxplot(activity_data, labels=activity_labels)
                ax.set_title(f'Outliers en {feature}')
                ax.set_ylabel('Valor')
                
                
                if len(activity_labels) > 3:
                    ax.tick_params(axis='x', rotation=45)
        
        
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / "outlier_boxplots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Boxplots de outliers guardados: {plot_path}")
    
    def activity_comparison(self):
        """
        Compara características entre diferentes actividades.
        """
        print("\n=== Comparación entre actividades ===")
        
        
        anova_results = {}
        
        for feature in self.feature_columns:
            groups = []
            for activity in self.df[self.activity_column].unique():
                group_data = self.df[self.df[self.activity_column] == activity][feature].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            
            if len(groups) > 1:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results[feature] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                except:
                    anova_results[feature] = {
                        'f_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
        
        
        anova_df = pd.DataFrame(anova_results).T
        anova_df = anova_df.sort_values('f_statistic', ascending=False)
        
        
        anova_path = self.output_dir / "statistics" / "anova_results.csv"
        anova_df.to_csv(anova_path)
        print(f"Resultados ANOVA guardados: {anova_path}")
        
        
        significant_features = anova_df[anova_df['significant'] == True]
        print(f"Características significativamente diferentes entre actividades: {len(significant_features)}")
        
        if len(significant_features) > 0:
            print("Top 10 características más discriminantes:")
            print(significant_features.head(10)[['f_statistic', 'p_value']].to_string())
            
            
            self._plot_activity_comparison(significant_features.head(8).index.tolist())
        
        self.analysis_results['activity_comparison'] = {
            'total_features_tested': len(anova_results),
            'significant_features': len(significant_features),
            'most_discriminant_features': significant_features.head(10).index.tolist() if len(significant_features) > 0 else []
        }
    
    def _plot_activity_comparison(self, features):
        """
        Crea gráficos de violín para comparar actividades.
        
        Args:
            features (list): Lista de características a comparar
        """
        n_cols = 2
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                
                sns.violinplot(
                    data=self.df,
                    x=self.activity_column,
                    y=feature,
                    ax=ax
                )
                
                ax.set_title(f'Distribución de {feature} por Actividad')
                ax.set_xlabel('Actividad')
                ax.set_ylabel('Valor')
                
                
                if len(self.df[self.activity_column].unique()) > 3:
                    ax.tick_params(axis='x', rotation=45)
        
        
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / "activity_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparación entre actividades guardada: {plot_path}")
    
    def dimensionality_analysis(self):
        """
        Realiza análisis de dimensionalidad con PCA y t-SNE.
        """
        print("\n=== Análisis de dimensionalidad ===")
        
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df[self.feature_columns].fillna(0))
        
        
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"Componentes principales para 95% de varianza: {n_components_95}")
        
        
        self._plot_pca_analysis(pca, X_pca, cumulative_variance)
        
        
        if len(self.df) > 100:
            print("Ejecutando t-SNE...")
            
            n_components_tsne = min(50, n_components_95)
            X_pca_reduced = X_pca[:, :n_components_tsne]
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df)//4))
            X_tsne = tsne.fit_transform(X_pca_reduced)
            
            self._plot_tsne(X_tsne)
        
        self.analysis_results['dimensionality_analysis'] = {
            'total_features': len(self.feature_columns),
            'components_for_95_variance': n_components_95,
            'variance_explained_by_first_10_components': cumulative_variance[:10].tolist()
        }
    
    def _plot_pca_analysis(self, pca, X_pca, cumulative_variance):
        """
        Crea gráficos relacionados con PCA.
        
        Args:
            pca: Objeto PCA ajustado
            X_pca: Datos transformados por PCA
            cumulative_variance: Varianza explicada acumulada
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        
        axes[0, 0].bar(range(1, min(21, len(pca.explained_variance_ratio_)+1)), 
                       pca.explained_variance_ratio_[:20])
        axes[0, 0].set_title('Varianza Explicada por Componente Principal')
        axes[0, 0].set_xlabel('Componente Principal')
        axes[0, 0].set_ylabel('Varianza Explicada')
        
        
        axes[0, 1].plot(range(1, min(51, len(cumulative_variance)+1)), 
                       cumulative_variance[:50])
        axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95%')
        axes[0, 1].set_title('Varianza Explicada Acumulada')
        axes[0, 1].set_xlabel('Número de Componentes')
        axes[0, 1].set_ylabel('Varianza Explicada Acumulada')
        axes[0, 1].legend()
        
        
        scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=pd.Categorical(self.df[self.activity_column]).codes,
                                   cmap='tab10', alpha=0.6)
        axes[1, 0].set_title('Proyección PCA (PC1 vs PC2)')
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        
        
        activities = self.df[self.activity_column].unique()
        for i, activity in enumerate(activities):
            axes[1, 0].scatter([], [], c=plt.cm.tab10(i), label=activity)
        axes[1, 0].legend()
        
        
        if X_pca.shape[1] > 2:
            scatter = axes[1, 1].scatter(X_pca[:, 1], X_pca[:, 2], 
                                       c=pd.Categorical(self.df[self.activity_column]).codes,
                                       cmap='tab10', alpha=0.6)
            axes[1, 1].set_title('Proyección PCA (PC2 vs PC3)')
            axes[1, 1].set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
            axes[1, 1].set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} varianza)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No hay suficientes componentes', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / "pca_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Análisis PCA guardado: {plot_path}")
    
    def _plot_tsne(self, X_tsne):
        """
        Crea gráfico de t-SNE.
        
        Args:
            X_tsne: Datos transformados por t-SNE
        """
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                            c=pd.Categorical(self.df[self.activity_column]).codes,
                            cmap='tab10', alpha=0.6)
        
        plt.title('Visualización t-SNE')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        
        activities = self.df[self.activity_column].unique()
        for i, activity in enumerate(activities):
            plt.scatter([], [], c=plt.cm.tab10(i), label=activity)
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "plots" / "tsne_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualización t-SNE guardada: {plot_path}")
    
    def data_quality_assessment(self):
        """
        Evalúa la calidad de los datos.
        """
        print("\n=== Evaluación de calidad de datos ===")
        
        quality_report = {}
        
        
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        quality_report['missing_data'] = {
            'total_missing_values': int(missing_data.sum()),
            'features_with_missing': int(len(missing_data[missing_data > 0])),
            'percentage_missing': float(missing_percentage.mean()),
            'worst_features': {k: float(v) for k, v in missing_percentage.nlargest(10).to_dict().items()}
        }
        
        
        duplicates = self.df.duplicated()
        quality_report['duplicates'] = {
            'total_duplicates': int(duplicates.sum()),
            'percentage_duplicates': float((duplicates.sum() / len(self.df)) * 100)
        }
        
        
        constant_features = []
        for feature in self.feature_columns:
            if self.df[feature].nunique() <= 1:
                constant_features.append(feature)
        
        quality_report['constant_features'] = {
            'count': len(constant_features),
            'features': constant_features
        }
        
        
        class_distribution = self.df[self.activity_column].value_counts()
        class_imbalance = class_distribution.max() / class_distribution.min()
        
        quality_report['class_distribution'] = {
            'class_counts': {k: int(v) for k, v in class_distribution.to_dict().items()},
            'imbalance_ratio': float(class_imbalance),
            'is_balanced': bool(class_imbalance < 3)  
        }
        
        
        quality_path = self.output_dir / "statistics" / "data_quality_report.json"
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        print(f"Reporte de calidad guardado: {quality_path}")
        
        
        print(f"Valores faltantes: {quality_report['missing_data']['total_missing_values']}")
        print(f"Duplicados: {quality_report['duplicates']['total_duplicates']}")
        print(f"Características constantes: {quality_report['constant_features']['count']}")
        print(f"Desbalance de clases: {class_imbalance:.2f}")
        
        self.analysis_results['data_quality'] = quality_report
    
    def generate_analysis_report(self):
        """
        Genera un reporte completo del análisis.
        """
        print("\n=== Generando reporte de análisis ===")
        
        
        full_report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'data_source': str(self.data_dir),
                'analyst': 'DataAnalyzer',
                'version': '1.0'
            },
            'analysis_results': self.analysis_results
        }
        
        
        report_path = self.output_dir / "reports" / "analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        
        text_report_path = self.output_dir / "reports" / "analysis_report.txt"
        with open(text_report_path, 'w') as f:
            f.write("REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Datos analizados: {self.data_dir}\n\n")
            
            
            if 'basic_info' in self.analysis_results:
                info = self.analysis_results['basic_info']
                f.write("INFORMACIÓN BÁSICA DEL DATASET:\n")
                f.write(f"- Total de muestras: {info['total_samples']}\n")
                f.write(f"- Total de características: {info['total_features']}\n")
                f.write(f"- Actividades: {', '.join(info['activities'])}\n")
                f.write("- Muestras por actividad:\n")
                for activity, count in info['samples_per_activity'].items():
                    f.write(f"  * {activity}: {count}\n")
                f.write("\n")
            
            
            if 'data_quality' in self.analysis_results:
                quality = self.analysis_results['data_quality']
                f.write("CALIDAD DE DATOS:\n")
                f.write(f"- Valores faltantes: {quality['missing_data']['total_missing_values']}\n")
                f.write(f"- Registros duplicados: {quality['duplicates']['total_duplicates']}\n")
                f.write(f"- Características constantes: {quality['constant_features']['count']}\n")
                f.write(f"- Ratio de desbalance: {quality['class_distribution']['imbalance_ratio']:.2f}\n")
                f.write(f"- Dataset balanceado: {'Sí' if quality['class_distribution']['is_balanced'] else 'No'}\n\n")
            
            
            if 'correlation_analysis' in self.analysis_results:
                corr = self.analysis_results['correlation_analysis']
                f.write("ANÁLISIS DE CORRELACIONES:\n")
                f.write(f"- Correlaciones altas encontradas: {corr['high_correlation_count']}\n")
                f.write(f"- Umbral usado: {corr['correlation_threshold']}\n\n")
            
            
            if 'activity_comparison' in self.analysis_results:
                comp = self.analysis_results['activity_comparison']
                f.write("COMPARACIÓN ENTRE ACTIVIDADES:\n")
                f.write(f"- Características testadas: {comp['total_features_tested']}\n")
                f.write(f"- Características significativas: {comp['significant_features']}\n")
                if comp['most_discriminant_features']:
                    f.write("- Características más discriminantes:\n")
                    for feature in comp['most_discriminant_features'][:5]:
                        f.write(f"  * {feature}\n")
                f.write("\n")
            
            
            if 'dimensionality_analysis' in self.analysis_results:
                dim = self.analysis_results['dimensionality_analysis']
                f.write("ANÁLISIS DE DIMENSIONALIDAD:\n")
                f.write(f"- Características originales: {dim['total_features']}\n")
                f.write(f"- Componentes para 95% varianza: {dim['components_for_95_variance']}\n\n")
            
            f.write("ARCHIVOS GENERADOS:\n")
            f.write("- Gráficos: analysis/plots/\n")
            f.write("- Estadísticas: analysis/statistics/\n")
            f.write("- Reportes: analysis/reports/\n")
        
        print(f"Reporte completo guardado: {report_path}")
        print(f"Reporte en texto guardado: {text_report_path}")
        
        return full_report


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Análisis exploratorio de datos para sistema de anotación")
    parser.add_argument("--data_dir", default="data/training_data", help="Directorio con los datos")
    parser.add_argument("--output_dir", default="analysis", help="Directorio de salida para análisis")
    parser.add_argument("--data_file", help="Archivo específico de datos a analizar")
    
    args = parser.parse_args()
    
    
    analyzer = DataAnalyzer(args.data_dir, args.output_dir)
    
    try:
        
        analyzer.load_data(args.data_file)
        
        
        print("Iniciando análisis exploratorio...")
        
        analyzer.descriptive_statistics()
        analyzer.correlation_analysis()
        analyzer.outlier_analysis()
        analyzer.activity_comparison()
        analyzer.dimensionality_analysis()
        analyzer.data_quality_assessment()
        
        
        analyzer.generate_analysis_report()
        
        print(f"\n¡Análisis completado exitosamente!")
        print(f"Resultados disponibles en: {args.output_dir}")
        print("\nArchivos principales:")
        print("- analysis/reports/analysis_report.txt")
        print("- analysis/plots/ (visualizaciones)")
        print("- analysis/statistics/ (datos estadísticos)")
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 