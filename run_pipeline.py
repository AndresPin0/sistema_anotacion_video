#!/usr/bin/env python3
"""
Pipeline Completo del Sistema de Anotación de Video
===================================================

Este script ejecuta el pipeline completo del proyecto:
1. Extracción de características de videos
2. Análisis exploratorio de datos
3. Entrenamiento de modelos de clasificación

Uso:
    python run_pipeline.py
    python run_pipeline.py --video_dir videos --quick_run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
from datetime import datetime

class PipelineRunner:
    """
    Ejecutor del pipeline completo del sistema de anotación de video.
    """
    
    def __init__(self, video_dir="videos", data_dir="data/training_data", 
                 models_dir="models", analysis_dir="analysis"):
        """
        Inicializa el ejecutor del pipeline.
        
        Args:
            video_dir (str): Directorio con videos de entrenamiento
            data_dir (str): Directorio para datos procesados
            models_dir (str): Directorio para modelos entrenados
            analysis_dir (str): Directorio para análisis de datos
        """
        self.video_dir = Path(video_dir)
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.analysis_dir = Path(analysis_dir)
        
        # Estado del pipeline
        self.steps_completed = []
        self.start_time = None
        
    def check_prerequisites(self):
        """
        Verifica que todos los prerequisitos estén disponibles.
        """
        print("=== Verificando prerequisitos ===")
        
        # Verificar que existe el directorio de videos
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Directorio de videos no encontrado: {self.video_dir}")
        
        # Verificar que hay videos en las carpetas de actividades
        activity_folders = ["caminar_hacia", "caminar_regreso", "girar_90", 
                           "girar_180", "sentarse", "ponerse_de_pie"]
        
        videos_found = 0
        for folder in activity_folders:
            folder_path = self.video_dir / folder
            if folder_path.exists():
                video_files = list(folder_path.glob("*.mp4")) + list(folder_path.glob("*.avi"))
                videos_found += len(video_files)
                print(f"  - {folder}: {len(video_files)} videos")
        
        if videos_found == 0:
            raise ValueError("No se encontraron videos en las carpetas de actividades")
        
        print(f"Total de videos encontrados: {videos_found}")
        
        # Verificar que los scripts principales existen (en su nueva ubicación)
        required_scripts = [
            "src/extract_features/extract_video_features.py", 
            "src/train_classifier/train_classifier.py", 
            "src/data_analysis/data_analysis.py"
        ]
        for script in required_scripts:
            if not Path(script).exists():
                raise FileNotFoundError(f"Script requerido no encontrado: {script}")
        
        print("✓ Todos los prerequisitos verificados")
        
    def run_feature_extraction(self, max_videos=None, max_frames=None):
        """
        Ejecuta la extracción de características de videos.
        
        Args:
            max_videos (int): Máximo número de videos por actividad (None para todos)
            max_frames (int): Máximo número de frames por video (None para todos)
        """
        print("\n=== Paso 1: Extracción de Características ===")
        
        cmd = [
            sys.executable, "src/extract_features/extract_video_features.py",
            "--video_dir", str(self.video_dir),
            "--output_dir", str(self.data_dir)
        ]
        
        if max_videos:
            cmd.extend(["--max_videos", str(max_videos)])
        if max_frames:
            cmd.extend(["--max_frames", str(max_frames)])
        
        print(f"Ejecutando: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            print("ERROR en extracción de características:")
            print(result.stderr)
            raise RuntimeError("Falló la extracción de características")
        
        print(f"✓ Extracción completada en {end_time - start_time:.1f}s")
        self.steps_completed.append("feature_extraction")
        
        # Mostrar resumen de la salida
        print("\nResumen de extracción:")
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if "Videos procesados:" in line or "Total de muestras" in line or "RESUMEN" in line:
                print(f"  {line}")
    
    def run_data_analysis(self):
        """
        Ejecuta el análisis exploratorio de datos.
        """
        print("\n=== Paso 2: Análisis Exploratorio de Datos ===")
        
        cmd = [
            sys.executable, "src/data_analysis/data_analysis.py",
            "--data_dir", str(self.data_dir),
            "--output_dir", str(self.analysis_dir)
        ]
        
        print(f"Ejecutando: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            print("ERROR en análisis de datos:")
            print(result.stderr)
            raise RuntimeError("Falló el análisis de datos")
        
        print(f"✓ Análisis completado en {end_time - start_time:.1f}s")
        self.steps_completed.append("data_analysis")
        
        # Mostrar resumen
        print("\nAnálisis generado:")
        print(f"  - Reportes: {self.analysis_dir}/reports/")
        print(f"  - Gráficos: {self.analysis_dir}/plots/")
        print(f"  - Estadísticas: {self.analysis_dir}/statistics/")
    
    def run_model_training(self, quick_run=False, n_features=50):
        """
        Ejecuta el entrenamiento de modelos.
        
        Args:
            quick_run (bool): Si ejecutar entrenamiento rápido (sin Grid Search)
            n_features (int): Número de características a seleccionar
        """
        print("\n=== Paso 3: Entrenamiento de Modelos ===")
        
        cmd = [
            sys.executable, "src/train_classifier/train_classifier.py",
            "--data_dir", str(self.data_dir),
            "--output_dir", str(self.models_dir),
            "--n_features", str(n_features)
        ]
        
        if quick_run:
            cmd.append("--no_grid_search")
            print("Modo rápido activado (sin Grid Search)")
        
        print(f"Ejecutando: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            print("ERROR en entrenamiento de modelos:")
            print(result.stderr)
            raise RuntimeError("Falló el entrenamiento de modelos")
        
        print(f"✓ Entrenamiento completado en {end_time - start_time:.1f}s")
        self.steps_completed.append("model_training")
        
        # Mostrar resumen de resultados
        print("\nModelos entrenados:")
        print(f"  - Modelos: {self.models_dir}/trained_models/")
        print(f"  - Evaluación: {self.models_dir}/evaluation/")
        print(f"  - Gráficos: {self.models_dir}/plots/")
        
        # Intentar mostrar el mejor modelo
        best_model_file = self.models_dir / "evaluation" / "best_model_info.json"
        if best_model_file.exists():
            import json
            try:
                with open(best_model_file, 'r') as f:
                    best_model_info = json.load(f)
                print(f"\nMejor modelo: {best_model_info['best_model']}")
                print(f"F1-Score: {best_model_info['metrics']['F1-Score']:.4f}")
            except:
                pass
    
    def run_complete_pipeline(self, quick_run=False, max_videos=None, max_frames=None):
        """
        Ejecuta el pipeline completo.
        
        Args:
            quick_run (bool): Si ejecutar en modo rápido
            max_videos (int): Máximo videos por actividad para pruebas
            max_frames (int): Máximo frames por video para pruebas
        """
        print("INICIANDO PIPELINE COMPLETO DEL SISTEMA DE ANOTACIÓN DE VIDEO")
        print("=" * 65)
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Modo: {'Rápido' if quick_run else 'Completo'}")
        
        if max_videos:
            print(f"Límite de videos por actividad: {max_videos}")
        if max_frames:
            print(f"Límite de frames por video: {max_frames}")
        
        self.start_time = time.time()
        
        try:
            # Verificar prerequisitos
            self.check_prerequisites()
            
            # Paso 1: Extracción de características
            self.run_feature_extraction(max_videos, max_frames)
            
            # Paso 2: Análisis de datos
            self.run_data_analysis()
            
            # Paso 3: Entrenamiento de modelos
            self.run_model_training(quick_run=quick_run)
            
            # Resumen final
            self.print_final_summary()
            
        except Exception as e:
            print(f"\n❌ ERROR EN EL PIPELINE: {str(e)}")
            print(f"Pasos completados: {', '.join(self.steps_completed)}")
            return False
        
        return True
    
    def print_final_summary(self):
        """
        Imprime un resumen final del pipeline ejecutado.
        """
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 65)
        print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 65)
        print(f"Tiempo total: {total_time:.1f}s ({total_time/60:.1f} minutos)")
        print(f"Pasos completados: {len(self.steps_completed)}/3")
        print(f"Finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📁 ARCHIVOS GENERADOS:")
        print(f"  • Datos de entrenamiento: {self.data_dir}/")
        print(f"  • Modelos entrenados: {self.models_dir}/")
        print(f"  • Análisis de datos: {self.analysis_dir}/")
        
        print("\n🚀 PRÓXIMOS PASOS:")
        print("  1. Revisar reportes de análisis en: analysis/reports/")
        print("  2. Revisar métricas de modelos en: models/evaluation/")
        print("  3. Ejecutar aplicación principal: python src/main.py")
        print("  4. Usar el mejor modelo para predicciones en tiempo real")
        
        print("\n📊 ARCHIVOS CLAVE:")
        print(f"  • Reporte de análisis: {self.analysis_dir}/reports/analysis_report.txt")
        print(f"  • Comparación de modelos: {self.models_dir}/evaluation/model_comparison.csv")
        print(f"  • Mejor modelo: {self.models_dir}/evaluation/best_model_info.json")
        print(f"  • Dataset completo: {self.data_dir}/complete_dataset.csv")


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Pipeline completo del sistema de anotación de video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    
    # Ejecutar pipeline completo
    python run_pipeline.py
    
    # Ejecutar en modo rápido (sin Grid Search)
    python run_pipeline.py --quick_run
    
    # Ejecutar con límites para pruebas
    python run_pipeline.py --max_videos 2 --max_frames 100 --quick_run
    
    # Especificar directorios personalizados
    python run_pipeline.py --video_dir mi_videos --models_dir mis_modelos
        """
    )
    
    parser.add_argument("--video_dir", default="videos", 
                       help="Directorio con videos de entrenamiento")
    parser.add_argument("--data_dir", default="data/training_data",
                       help="Directorio para datos procesados")
    parser.add_argument("--models_dir", default="models",
                       help="Directorio para modelos entrenados")
    parser.add_argument("--analysis_dir", default="analysis",
                       help="Directorio para análisis de datos")
    
    parser.add_argument("--quick_run", action="store_true",
                       help="Ejecutar en modo rápido (sin Grid Search)")
    parser.add_argument("--max_videos", type=int,
                       help="Máximo número de videos por actividad (para pruebas)")
    parser.add_argument("--max_frames", type=int,
                       help="Máximo número de frames por video (para pruebas)")
    
    # Pasos individuales
    parser.add_argument("--only_extract", action="store_true",
                       help="Solo ejecutar extracción de características")
    parser.add_argument("--only_analyze", action="store_true",
                       help="Solo ejecutar análisis de datos")
    parser.add_argument("--only_train", action="store_true",
                       help="Solo ejecutar entrenamiento de modelos")
    
    args = parser.parse_args()
    
    # Crear ejecutor del pipeline
    pipeline = PipelineRunner(
        video_dir=args.video_dir,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        analysis_dir=args.analysis_dir
    )
    
    try:
        # Verificar prerequisitos básicos
        pipeline.check_prerequisites()
        
        # Ejecutar pasos individuales si se especifica
        if args.only_extract:
            pipeline.run_feature_extraction(args.max_videos, args.max_frames)
        elif args.only_analyze:
            pipeline.run_data_analysis()
        elif args.only_train:
            pipeline.run_model_training(args.quick_run)
        else:
            # Ejecutar pipeline completo
            success = pipeline.run_complete_pipeline(
                quick_run=args.quick_run,
                max_videos=args.max_videos,
                max_frames=args.max_frames
            )
            
            if not success:
                sys.exit(1)
        
        print("\n✅ Ejecución finalizada correctamente")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 