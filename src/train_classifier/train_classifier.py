
"""
Entrenador de Modelos de Clasificación
=====================================

Este script entrena diferentes modelos de machine learning para clasificar
actividades humanas basándose en las características extraídas de videos.

Funcionalidades:
- Entrena múltiples algoritmos (SVM, Random Forest, XGBoost, etc.)
- Optimización de hiperparámetros con Grid Search
- Validación cruzada y métricas de evaluación
- Análisis de importancia de características
- Guarda modelos entrenados y métricas de rendimiento

Uso:
    python train_classifier.py
    python train_classifier.py --data_dir data/training_data --output_dir models
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE


try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Advertencia: XGBoost no está disponible. Se excluirá del entrenamiento.")

class ActivityClassifierTrainer:
    """
    Entrenador de modelos de clasificación de actividades.
    """
    
    def __init__(self, data_dir="data/training_data", output_dir="models"):
        """
        Inicializa el entrenador.
        
        Args:
            data_dir (str): Directorio con los datos de entrenamiento
            output_dir (str): Directorio donde guardar modelos y resultados
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        (self.output_dir / "trained_models").mkdir(exist_ok=True)
        (self.output_dir / "evaluation").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "feature_analysis").mkdir(exist_ok=True)
        
        
        self.models = self._setup_models()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.class_names = None
        
        self.trained_models = {}
        self.evaluation_results = {}
        
    def _setup_models(self):
        """
        Configura los modelos a entrenar con sus hiperparámetros.
        
        Returns:
            dict: Diccionario con modelos y parámetros para Grid Search
        """
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'l1_ratio': [0.1, 0.5, 0.9]  
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            }
        }
        
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        
        return models
    
    def load_data(self, file_path=None):
        """
        Carga los datos de entrenamiento.
        
        Args:
            file_path (str): Ruta específica al archivo de datos. Si es None, busca automáticamente.
        """
        print("=== Cargando datos de entrenamiento ===")
        
        if file_path is None:
            
            possible_files = [
                self.data_dir / "processed" / "training_features.csv",
                self.data_dir / "complete_dataset.csv"
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    break
            else:
                raise FileNotFoundError(f"No se encontraron datos de entrenamiento en {self.data_dir}")
        
        print(f"Cargando datos desde: {file_path}")
        
        
        df = pd.read_csv(file_path)
        print(f"Datos cargados: {df.shape[0]} muestras, {df.shape[1]} columnas")
        
        
        if 'activity' not in df.columns:
            raise ValueError("El dataset debe contener una columna 'activity'")
        
        
        feature_columns = [col for col in df.columns 
                          if col not in ['activity', 'video_path', 'frame_idx', 'timestamp']]
        
        X = df[feature_columns]
        y = df['activity']
        
        print(f"Características: {len(feature_columns)}")
        print(f"Clases encontradas: {y.nunique()}")
        print("Distribución de clases:")
        print(y.value_counts())
        
        
        print(f"\nValores faltantes por columna:")
        missing_counts = X.isnull().sum()
        missing_columns = missing_counts[missing_counts > 0]
        
        if len(missing_columns) > 0:
            print(missing_columns)
            
            X = X.fillna(X.median())
            print("Valores faltantes rellenados con la mediana")
        else:
            print("No hay valores faltantes")
        
        
        self.feature_names = feature_columns
        self.class_names = sorted(y.unique())
        
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nDatos divididos:")
        print(f"Entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"Prueba: {self.X_test.shape[0]} muestras")
        
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("Características escaladas con StandardScaler")
        
        
        scaler_path = self.output_dir / "trained_models" / "scaler.joblib"
        encoder_path = self.output_dir / "trained_models" / "label_encoder.joblib"
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Scaler guardado: {scaler_path}")
        print(f"Label encoder guardado: {encoder_path}")
    
    def feature_selection(self, n_features=50):
        """
        Realiza selección de características.
        
        Args:
            n_features (int): Número de características a seleccionar
        """
        print(f"\n=== Selección de características (top {n_features}) ===")
        
        
        selector = SelectKBest(score_func=f_classif, k=min(n_features, len(self.feature_names)))
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_selected = selector.transform(self.X_test_scaled)
        
        
        selected_features = selector.get_support(indices=True)
        selected_feature_names = [self.feature_names[i] for i in selected_features]
        feature_scores = selector.scores_[selected_features]
        
        print(f"Características seleccionadas: {len(selected_feature_names)}")
        
        
        feature_info = {
            'selected_features': selected_feature_names,
            'feature_scores': feature_scores.tolist(),
            'selection_method': 'SelectKBest_f_classif'
        }
        
        feature_info_path = self.output_dir / "feature_analysis" / "selected_features.json"
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"Información de características guardada: {feature_info_path}")
        
        
        self.X_train_scaled = X_train_selected
        self.X_test_scaled = X_test_selected
        self.feature_names = selected_feature_names
        
        
        self._plot_feature_importance(selected_feature_names, feature_scores, "Feature Selection Scores")
    
    def train_models(self, use_grid_search=True, cv_folds=5):
        """
        Entrena todos los modelos configurados.
        
        Args:
            use_grid_search (bool): Si usar Grid Search para optimización de hiperparámetros
            cv_folds (int): Número de folds para validación cruzada
        """
        print(f"\n=== Entrenando modelos (Grid Search: {use_grid_search}) ===")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model_config in self.models.items():
            print(f"\n--- Entrenando {model_name} ---")
            
            try:
                if use_grid_search:
                    
                    grid_search = GridSearchCV(
                        model_config['model'],
                        model_config['params'],
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    grid_search.fit(self.X_train_scaled, self.y_train)
                    best_model = grid_search.best_estimator_
                    
                    print(f"Mejores parámetros: {grid_search.best_params_}")
                    print(f"Mejor score CV: {grid_search.best_score_:.4f}")
                    
                else:
                    
                    best_model = model_config['model']
                    best_model.fit(self.X_train_scaled, self.y_train)
                
                
                self.trained_models[model_name] = best_model
                
                
                train_score = best_model.score(self.X_train_scaled, self.y_train)
                test_score = best_model.score(self.X_test_scaled, self.y_test)
                
                
                cv_scores = cross_val_score(best_model, self.X_train_scaled, self.y_train, cv=cv)
                
                print(f"Score entrenamiento: {train_score:.4f}")
                print(f"Score prueba: {test_score:.4f}")
                print(f"CV Score medio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                
                model_path = self.output_dir / "trained_models" / f"{model_name}.joblib"
                joblib.dump(best_model, model_path)
                print(f"Modelo guardado: {model_path}")
                
            except Exception as e:
                print(f"Error entrenando {model_name}: {str(e)}")
                continue
    
    def evaluate_models(self):
        """
        Evalúa todos los modelos entrenados con métricas detalladas.
        """
        print("\n=== Evaluando modelos ===")
        
        for model_name, model in self.trained_models.items():
            print(f"\n--- Evaluación de {model_name} ---")
            
            
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = None
            
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(self.X_test_scaled)
            
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                self.y_test, y_pred, average='weighted'
            )
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            
            class_report = classification_report(
                self.y_test, y_pred,
                target_names=self.class_names,
                output_dict=True
            )
            
            
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            
            self.evaluation_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': y_pred.tolist(),
                'true_labels': self.y_test.tolist()
            }
            
            if y_pred_proba is not None:
                self.evaluation_results[model_name]['prediction_probabilities'] = y_pred_proba.tolist()
            
            
            self._plot_confusion_matrix(conf_matrix, model_name)
            
            
            if hasattr(model, 'feature_importances_'):
                self._plot_feature_importance(
                    self.feature_names, 
                    model.feature_importances_, 
                    f"{model_name} Feature Importance"
                )
        
        
        evaluation_path = self.output_dir / "evaluation" / "model_evaluation_results.json"
        with open(evaluation_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"\nResultados de evaluación guardados: {evaluation_path}")
        
        
        self._create_model_comparison()
    
    def _plot_confusion_matrix(self, conf_matrix, model_name):
        """
        Crea y guarda la matriz de confusión.
        
        Args:
            conf_matrix (array): Matriz de confusión
            model_name (str): Nombre del modelo
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicho')
        plt.ylabel('Real')
        plt.tight_layout()
        
        plot_path = self.output_dir / "plots" / f"confusion_matrix_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matriz de confusión guardada: {plot_path}")
    
    def _plot_feature_importance(self, feature_names, importance_scores, title):
        """
        Crea gráfico de importancia de características.
        
        Args:
            feature_names (list): Nombres de las características
            importance_scores (array): Scores de importancia
            title (str): Título del gráfico
        """
        
        indices = np.argsort(importance_scores)[::-1]
        
        
        top_n = min(20, len(feature_names))
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(top_n), importance_scores[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.tight_layout()
        
        
        file_name = title.lower().replace(' ', '_').replace('-', '_') + '.png'
        plot_path = self.output_dir / "plots" / file_name
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de importancia guardado: {plot_path}")
    
    def _create_model_comparison(self):
        """
        Crea una comparación visual de todos los modelos.
        """
        if not self.evaluation_results:
            return
        
        
        models = list(self.evaluation_results.keys())
        accuracies = [self.evaluation_results[m]['accuracy'] for m in models]
        precisions = [self.evaluation_results[m]['precision'] for m in models]
        recalls = [self.evaluation_results[m]['recall'] for m in models]
        f1_scores = [self.evaluation_results[m]['f1_score'] for m in models]
        
        
        comparison_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        })
        
        
        comparison_path = self.output_dir / "evaluation" / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Modelos', fontsize=16)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            bars = ax.bar(models, comparison_df[metric])
            ax.set_title(f'{metric} por Modelo')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            
            if len(models) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        comparison_plot_path = self.output_dir / "plots" / "model_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparación de modelos guardada: {comparison_path}")
        print(f"Gráfico de comparación guardado: {comparison_plot_path}")
        
        
        print("\n=== RESUMEN DE MODELOS ===")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        
        best_model_idx = comparison_df['F1-Score'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]['Model']
        best_f1 = comparison_df.iloc[best_model_idx]['F1-Score']
        
        print(f"\nMejor modelo: {best_model} (F1-Score: {best_f1:.4f})")
        
        
        best_model_info = {
            'best_model': best_model,
            'metrics': comparison_df.iloc[best_model_idx].to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        best_model_path = self.output_dir / "evaluation" / "best_model_info.json"
        with open(best_model_path, 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        print(f"Información del mejor modelo guardada: {best_model_path}")
    
    def generate_training_report(self):
        """
        Genera un reporte completo del entrenamiento.
        """
        print("\n=== Generando reporte de entrenamiento ===")
        
        report = {
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'data_source': str(self.data_dir),
                'total_samples': len(self.y_train) + len(self.y_test),
                'training_samples': len(self.y_train),
                'test_samples': len(self.y_test),
                'num_features': len(self.feature_names),
                'num_classes': len(self.class_names),
                'classes': self.class_names
            },
            'models_trained': list(self.trained_models.keys()),
            'evaluation_summary': {}
        }
        
        
        for model_name, results in self.evaluation_results.items():
            report['evaluation_summary'][model_name] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }
        
        
        report_path = self.output_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Reporte de entrenamiento guardado: {report_path}")
        
        
        text_report_path = self.output_dir / "training_report.txt"
        with open(text_report_path, 'w') as f:
            f.write("REPORTE DE ENTRENAMIENTO - SISTEMA DE ANOTACIÓN DE VIDEO\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directorio de datos: {self.data_dir}\n")
            f.write(f"Directorio de salida: {self.output_dir}\n\n")
            
            f.write("INFORMACIÓN DEL DATASET:\n")
            f.write(f"- Total de muestras: {report['training_info']['total_samples']}\n")
            f.write(f"- Muestras de entrenamiento: {report['training_info']['training_samples']}\n")
            f.write(f"- Muestras de prueba: {report['training_info']['test_samples']}\n")
            f.write(f"- Número de características: {report['training_info']['num_features']}\n")
            f.write(f"- Número de clases: {report['training_info']['num_classes']}\n")
            f.write(f"- Clases: {', '.join(report['training_info']['classes'])}\n\n")
            
            f.write("MODELOS ENTRENADOS:\n")
            for model_name in report['models_trained']:
                f.write(f"- {model_name}\n")
            f.write("\n")
            
            f.write("RESULTADOS DE EVALUACIÓN:\n")
            f.write("-" * 40 + "\n")
            for model_name, metrics in report['evaluation_summary'].items():
                f.write(f"{model_name}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
        
        print(f"Reporte en texto guardado: {text_report_path}")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(description="Entrenador de modelos de clasificación de actividades")
    parser.add_argument("--data_dir", default="data/training_data", help="Directorio con datos de entrenamiento")
    parser.add_argument("--output_dir", default="models", help="Directorio de salida para modelos")
    parser.add_argument("--no_grid_search", action="store_true", help="No usar Grid Search (más rápido)")
    parser.add_argument("--cv_folds", type=int, default=5, help="Número de folds para validación cruzada")
    parser.add_argument("--n_features", type=int, default=50, help="Número de características a seleccionar")
    parser.add_argument("--data_file", help="Archivo específico de datos a usar")
    
    args = parser.parse_args()
    
    
    trainer = ActivityClassifierTrainer(args.data_dir, args.output_dir)
    
    try:
        
        trainer.load_data(args.data_file)
        
        
        trainer.feature_selection(args.n_features)
        
        
        trainer.train_models(
            use_grid_search=not args.no_grid_search,
            cv_folds=args.cv_folds
        )
        
        
        trainer.evaluate_models()
        
        
        trainer.generate_training_report()
        
        print(f"\n¡Entrenamiento completado exitosamente!")
        print(f"Modelos y resultados disponibles en: {args.output_dir}")
        print("\nPróximos pasos:")
        print("1. Revisar métricas en: models/evaluation/")
        print("2. Usar el mejor modelo en la aplicación principal")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 