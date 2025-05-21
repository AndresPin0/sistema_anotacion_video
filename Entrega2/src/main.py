import os
import sys
from Entrega2.src.gui.gui_app import main as start_gui
from Entrega2.src.utils.camera_selector import select_camera

if __name__ == "__main__":
    print("Sistema de Anotación de Video - Inicializando...")
    
    # Seleccionar cámara
    print("Seleccionando cámara...")
    camera_id = select_camera()
    print(f"Cámara seleccionada: ID {camera_id}")
    
    # Iniciar la interfaz gráfica con la cámara seleccionada
    start_gui(camera_id) 