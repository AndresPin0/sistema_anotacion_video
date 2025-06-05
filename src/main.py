
"""
Sistema de Anotación de Video - Punto de Entrada Principal
=========================================================

Este script inicia la aplicación principal del sistema de anotación de video.
Incluye selección automática de cámara y manejo de errores.

Uso:
    python src/main.py
    python src/main.py --camera_id 0
"""

import os
import sys
import argparse
from pathlib import Path


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def main(camera_id=None):
    """
    Función principal del sistema de anotación de video.
    
    Args:
        camera_id (int): ID de la cámara a usar (None para selección automática)
    """
    print("=" * 60)
    print("SISTEMA DE ANOTACIÓN DE VIDEO")
    print("Proyecto Final - Inteligencia Artificial 1")
    print("=" * 60)
    
    try:
        
        from src.gui.gui_app import main as start_gui
        from src.utils.camera_selector import select_camera
        
        print("✓ Módulos del sistema cargados correctamente")
        
        
        if camera_id is None:
            print("\nSeleccionando cámara...")
            camera_id = select_camera()
            
        print(f"✓ Cámara seleccionada: ID {camera_id}")
        
        
        models_dir = Path("models")
        if models_dir.exists() and (models_dir / "trained_models").exists():
            print("✓ Modelos entrenados encontrados")
        else:
            print("⚠️  No se encontraron modelos entrenados")
            print("   Para mejores resultados, ejecuta primero: python run_pipeline.py")
        
        print("\n🚀 Iniciando interfaz gráfica...")
        print("   - Usa la cámara para detectar actividades en tiempo real")
        print("   - Ajusta umbrales desde el menú si es necesario")
        print("   - Para cerrar, usa Ctrl+C o cierra la ventana")
        
        
        start_gui(camera_id)
        
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("\nVerifica que:")
        print("1. Las dependencias estén instaladas: pip install -r requirements.txt")
        print("2. Estés ejecutando desde el directorio raíz del proyecto")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        print("\nSi el problema persiste, revisa:")
        print("1. Que la cámara esté disponible y no en uso")
        print("2. Que tengas permisos para acceder a la cámara")
        print("3. Los logs de error en la consola")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de Anotación de Video - Aplicación Principal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    
    
    python src/main.py
    
    
    python src/main.py --camera_id 0
    
    
    python run_pipeline.py --quick_run
    python src/main.py
        """
    )
    
    parser.add_argument(
        "--camera_id", 
        type=int, 
        help="ID de la cámara a usar (0, 1, 2, etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        main(args.camera_id)
    except KeyboardInterrupt:
        print("\n\n👋 Sistema cerrado por el usuario")
        sys.exit(0) 