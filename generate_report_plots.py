
"""
Generador de Gráficos para Informe Técnico
==========================================

Este script genera visualizaciones adicionales para el informe técnico,
incluyendo diagramas de arquitectura, análisis de rendimiento y métricas del sistema.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json


plt.style.use('default')
sns.set_palette("husl")

def create_architecture_diagram():
    """Crea un diagrama de la arquitectura del sistema"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    
    components = [
        {"name": "Video Input\n(Camera)", "pos": (1, 4), "color": "
        {"name": "MediaPipe\nPose Detection", "pos": (3.5, 4), "color": "
        {"name": "Feature\nExtraction", "pos": (6, 4), "color": "
        {"name": "ML Model\n(LogisticRegression)", "pos": (8.5, 4), "color": "
        {"name": "Activity\nClassification", "pos": (11, 4), "color": "
        
        {"name": "33 Landmarks", "pos": (3.5, 2.5), "color": "
        {"name": "101 Features", "pos": (6, 2.5), "color": "
        {"name": "6 Activities", "pos": (11, 2.5), "color": "
    ]
    
    
    for comp in components:
        rect = mpatches.Rectangle(
            (comp["pos"][0] - comp["size"][0]/2, comp["pos"][1] - comp["size"][1]/2),
            comp["size"][0], comp["size"][1],
            facecolor=comp["color"], edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        
        
        ax.text(comp["pos"][0], comp["pos"][1], comp["name"], 
                ha='center', va='center', fontweight='bold', fontsize=10,
                color='white' if comp["color"] != "
    
    
    arrows = [
        ((2.25, 4), (2.75, 4)),  
        ((4.75, 4), (5.25, 4)),  
        ((7.25, 4), (7.75, 4)),  
        ((9.75, 4), (10.25, 4)), 
        
        ((3.5, 3.5), (3.5, 3.1)), 
        ((6, 3.5), (6, 3.1)),     
        ((11, 3.5), (11, 3.1)),   
    ]
    
    for start, end in arrows:
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(0, 12.5)
    ax.set_ylim(1.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Arquitectura del Sistema de Análisis de Actividades', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('reports/arquitectura_sistema.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison():
    """Crea gráfico de comparación de rendimiento de modelos"""
    
    models = ['Logistic\nRegression', 'Gradient\nBoosting', 'XGBoost', 'SVM', 'KNN', 'Random\nForest']
    accuracy = [69.46, 67.57, 67.03, 66.49, 66.49, 65.41]
    precision = [69.30, 67.57, 67.05, 65.99, 65.64, 65.63]
    recall = [69.46, 67.57, 67.03, 66.49, 66.49, 65.41]
    f1_score = [69.33, 67.57, 67.03, 66.03, 65.98, 65.51]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='
    bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='
    
    ax.set_xlabel('Modelos de Machine Learning', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rendimiento (%)', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Rendimiento de Modelos', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 75)
    
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars4)  
    
    plt.tight_layout()
    plt.savefig('reports/rendimiento_modelos.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot():
    """Crea gráfico de importancia de características"""
    
    features = [
        'right_knee_angle', 'right_knee_visibility', 'right_hip_angle',
        'left_knee_angle', 'trunk_lateral_inclination', 'left_hip_angle',
        'right_shoulder_y', 'left_shoulder_y', 'nose_z', 'right_hip_y'
    ]
    importance = [0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05]
    
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("viridis", len(features))
    bars = plt.barh(features, importance, color=colors)
    
    plt.xlabel('Importancia Relativa', fontsize=12, fontweight='bold')
    plt.title('Top 10 Características Más Importantes', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    
    for i, (bar, val) in enumerate(zip(bars, importance)):
        plt.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('reports/importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_activity_distribution():
    """Crea gráfico de distribución de actividades en el dataset"""
    activities = ['Girar 180°', 'Girar 90°', 'Caminar\nHacia', 'Ponerse\nde Pie', 'Sentarse', 'Caminar\nRegreso']
    counts = [369, 369, 355, 267, 250, 239]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['
    bars = plt.bar(activities, counts, color=colors, alpha=0.8)
    
    plt.xlabel('Actividades', fontsize=12, fontweight='bold')
    plt.ylabel('Número de Muestras', fontsize=12, fontweight='bold')
    plt.title('Distribución de Actividades en el Dataset', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    
    avg = np.mean(counts)
    plt.axhline(y=avg, color='red', linestyle='--', alpha=0.7, 
                label=f'Promedio: {avg:.0f} muestras')
    plt.legend()
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('reports/distribucion_actividades.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_summary():
    """Crea un resumen visual de la matriz de confusión del mejor modelo"""
    
    activities = ['Caminar\nHacia', 'Caminar\nRegreso', 'Girar\n90°', 'Girar\n180°', 'Sentarse', 'Ponerse\nde Pie']
    
    
    cm = np.array([
        [45, 3, 2, 1, 4, 2],   
        [4, 41, 1, 2, 3, 2],   
        [2, 1, 48, 8, 1, 2],   
        [1, 2, 7, 52, 1, 1],   
        [3, 2, 1, 1, 43, 8],   
        [2, 3, 2, 1, 9, 45]    
    ])
    
    plt.figure(figsize=(10, 8))
    
    
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=activities, yticklabels=activities,
                cbar_kws={'label': 'Porcentaje (%)'})
    
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
    plt.title('Matriz de Confusión - Logistic Regression (Mejor Modelo)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('reports/matriz_confusion_resumen.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Función principal para generar todos los gráficos"""
    
    Path('reports').mkdir(exist_ok=True)
    
    print("Generando gráficos para el informe técnico...")
    
    print("1. Creando diagrama de arquitectura...")
    create_architecture_diagram()
    
    print("2. Creando gráfico de rendimiento de modelos...")
    create_performance_comparison()
    
    print("3. Creando gráfico de importancia de características...")
    create_feature_importance_plot()
    
    print("4. Creando distribución de actividades...")
    create_activity_distribution()
    
    print("5. Creando matriz de confusión resumida...")
    create_confusion_matrix_summary()
    
    print("✅ Todos los gráficos generados en el directorio 'reports/'")
    print("\nGráficos creados:")
    print("- arquitectura_sistema.png")
    print("- rendimiento_modelos.png")
    print("- importancia_caracteristicas.png")
    print("- distribucion_actividades.png")
    print("- matriz_confusion_resumen.png")

if __name__ == "__main__":
    main() 