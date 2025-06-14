from app.gui_app import GestureApp
import os

if __name__ == "__main__":
    # Asegurar que las carpetas de datos existen
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Iniciar la aplicación
    app = GestureApp()
    app.mainloop()
