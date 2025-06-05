
"""
Script de Entrenamiento Mejorado
==============================

Este script implementa el entrenamiento del modelo usando XGBoost
con validación cruzada y optimización de hiperparámetros.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import optuna
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el dataset de características procesadas.
    
    Args:
        data_dir: Directorio con los datos procesados
    Returns:
        Tuple[np.ndarray, np.ndarray]: (características, etiquetas)
    """
    data_path = Path(data_dir)
    features = np.load(data_path / "features.npy")
    labels = np.load(data_path / "labels.npy")
    
    print(f"Dataset cargado: {features.shape[0]} muestras, {features.shape[1]} características")
    return features, labels

def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, 
                           n_trials: int = 100) -> Dict:
    """
    Optimiza hiperparámetros usando Optuna.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        n_trials: Número de pruebas de optimización
    Returns:
        Dict: Mejores hiperparámetros encontrados
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train))
        }
        
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                params: Dict, model_dir: str = "models/trained_models") -> None:
    """
    Entrena el modelo final con los mejores hiperparámetros.
    
    Args:
        X_train: Datos de entrenamiento
        y_train: Etiquetas de entrenamiento
        params: Hiperparámetros optimizados
        model_dir: Directorio para guardar el modelo
    """
    
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain)
    
    
    model.save_model(str(model_path / "xgboost_model.json"))
    
    
    with open(model_path / "hyperparameters.json", "w") as f:
        json.dump(params, f, indent=4)

def evaluate_model(model: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray,
                  class_names: List[str], output_dir: str = "reports") -> None:
    """
    Evalúa el modelo y genera visualizaciones.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        class_names: Nombres de las clases
        output_dir: Directorio para guardar reportes
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open(output_path / "classification_report.txt", "w") as f:
        f.write(report)
    
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()
    
    
    importance_scores = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame.from_dict(importance_scores, orient='index', 
                                         columns=['importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    importance_df.head(20).plot(kind='bar')
    plt.title('Top 20 Características Más Importantes')
    plt.xlabel('Característica')
    plt.ylabel('Importancia')
    plt.tight_layout()
    plt.savefig(output_path / "feature_importance.png")
    plt.close()

def main():
    """Función principal de entrenamiento."""
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO MEJORADO")
    print("=" * 60)
    
    
    print("\n1. Cargando dataset...")
    X, y = load_dataset()
    
    
    print("\n2. Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    print("\n3. Aplicando preprocesamiento...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    joblib.dump(scaler, "models/trained_models/scaler.joblib")
    
    
    print("\n4. Optimizando hiperparámetros...")
    best_params = optimize_hyperparameters(X_train_scaled, y_train)
    print(f"Mejores hiperparámetros encontrados: {best_params}")
    
    
    print("\n5. Entrenando modelo final...")
    train_model(X_train_scaled, y_train, best_params)
    
    
    print("\n6. Evaluando modelo...")
    model = xgb.Booster()
    model.load_model("models/trained_models/xgboost_model.json")
    
    class_names = [
        'caminar_hacia', 'caminar_regreso', 'girar_90',
        'girar_180', 'sentarse', 'ponerse_pie'
    ]
    
    evaluate_model(model, X_test_scaled, y_test, class_names)
    
    print("\n✅ Entrenamiento completado!")
    print("\nArchivos generados:")
    print("- models/trained_models/xgboost_model.json")
    print("- models/trained_models/scaler.joblib")
    print("- models/trained_models/hyperparameters.json")
    print("- reports/classification_report.txt")
    print("- reports/confusion_matrix.png")
    print("- reports/feature_importance.png")

if __name__ == "__main__":
    main() 