import customtkinter as ctk
from tkinter import messagebox
import os
import threading
from PIL import Image
import cv2
from customtkinter import CTkImage

from app.core_logic import (
    save_config,
    load_config,
    get_capture_status,
    cleanup_project_files,
    train_model,
    OpenCVController,
)


class GestureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Panel de Control de Gestos")
        self.geometry("400x520")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.opencv_controller = None
        self.video_window = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()  # Bloqueo para sincronización

        self.last_frame = None

        self.setup_widgets()
        self.update_button_states()

    def setup_widgets(self):
        # (Sin cambios aquí)
        # ... (código idéntico a la versión anterior) ...
        title_label = ctk.CTkLabel(
            self,
            text="Control de Juegos por Gestos",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title_label.pack(pady=(20, 10))
        self.status_label = ctk.CTkLabel(
            self,
            text="Bienvenido. Comienza configurando los jugadores.",
            wraplength=380,
        )
        self.status_label.pack(pady=10)
        self.btn_setup = ctk.CTkButton(
            self,
            text="1. Configurar Jugadores",
            command=self.open_setup_window,
            height=40,
        )
        self.btn_setup.pack(pady=8, padx=40, fill="x")
        self.btn_capture = ctk.CTkButton(
            self, text="2. Capturar Gestos", command=self.open_capture_window, height=40
        )
        self.btn_capture.pack(pady=8, padx=40, fill="x")
        self.btn_train = ctk.CTkButton(
            self, text="3. Entrenar Modelo", command=self.run_training, height=40
        )
        self.btn_train.pack(pady=8, padx=40, fill="x")
        self.btn_play = ctk.CTkButton(
            self,
            text="🚀 JUGAR 🚀",
            command=lambda: self.run_opencv_task("play"),
            height=50,
            fg_color="#2ECC71",
            hover_color="#27AE60",
        )
        self.btn_play.pack(pady=15, padx=40, fill="x")
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(pady=20, padx=40, fill="x", side="bottom")
        btn_clean = ctk.CTkButton(
            bottom_frame,
            text="Limpiar Todo",
            command=self.clean_and_reset,
            fg_color="#E74C3C",
            hover_color="#C0392B",
        )
        btn_clean.pack(side="left", expand=True, padx=(0, 5))
        btn_exit = ctk.CTkButton(bottom_frame, text="Salir", command=self.on_closing)
        btn_exit.pack(side="right", expand=True, padx=(5, 0))

    def run_opencv_task(self, mode, player_id=None, key=None):
        if self.video_window:
            return
        self.stop_event.clear()

        try:
            self.opencv_controller = OpenCVController(self.update_gui_from_cv)
        except IOError as e:
            messagebox.showerror("Error de Cámara", str(e))
            return

        self.video_window = ctk.CTkToplevel(self)
        self.video_window.protocol("WM_DELETE_WINDOW", self.stop_opencv_task)

        # Crear el label de video
        self.video_label = ctk.CTkLabel(self.video_window, text="")
        self.video_label.pack(padx=10, pady=10)

        # Crear widgets adicionales si es modo captura
        if mode == "capture":
            self.video_window.title(f"Capturando para Jugador {player_id}")
            self.capture_info_label = ctk.CTkLabel(
                self.video_window, text="Muestras: 0 / 0"
            )
            self.capture_info_label.pack(pady=(0, 5))
            self.progress_bar = ctk.CTkProgressBar(self.video_window)
            self.progress_bar.set(0)
            self.progress_bar.pack(pady=10, padx=10, fill="x")

            # Lanzar el hilo de captura
            args = (player_id, key, self.stop_event)
            target = self.opencv_controller.run_capture
            self.thread = threading.Thread(
                target=lambda: self.thread_wrapper(
                    target, args, on_complete=self.handle_capture_completion
                )
            )

        else:  # Modo juego
            self.video_window.title("Modo Juego")
            args = (self.stop_event,)
            target = self.opencv_controller.run_play
            self.thread = threading.Thread(
                target=lambda: self.thread_wrapper(target, args)
            )

        self.thread.start()
        # Iniciar el bucle de renderizado de la GUI
        self.render_loop()

    def thread_wrapper(self, target_func, args, on_complete=None):
        """Wrapper que ejecuta la función del hilo y llama a on_complete si se proporciona."""
        result = target_func(*args)
        if on_complete:
            self.after(0, lambda: on_complete(result))

    def render_loop(self):
        """Bucle principal de renderizado de la GUI."""
        if not self.stop_event.is_set():
            with self.frame_lock:
                if self.last_frame is not None:
                    # **OPTIMIZACIÓN**: Reusar el objeto CTkImage
                    image_pil = Image.fromarray(
                        cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
                    )
                    if not hasattr(self.video_label, "_ctk_image_obj"):
                        # Crear el objeto CTkImage la primera vez
                        h, w, _ = self.last_frame.shape
                        self.video_label._ctk_image_obj = CTkImage(
                            light_image=image_pil, size=(w, h)
                        )
                    else:
                        # Reconfigurar el objeto existente, mucho más rápido
                        self.video_label._ctk_image_obj.configure(light_image=image_pil)

                    self.video_label.configure(image=self.video_label._ctk_image_obj)

            self.after(16, self.render_loop)  # Aproximadamente 60 FPS

    def update_gui_from_cv(self, **kwargs):
        """Función llamada desde el hilo de OpenCV."""
        with self.frame_lock:
            self.last_frame = kwargs.get("image")

        # Actualizar widgets de progreso desde el hilo principal de la GUI
        if "progress" in kwargs:
            self.after(0, self.progress_bar.set, kwargs["progress"])
            self.after(
                0,
                self.capture_info_label.configure,
                {"text": f"Muestras: {kwargs['count']} / {kwargs['total']}"},
            )

    def handle_capture_completion(self, success):
        self.stop_opencv_task()
        if success:
            messagebox.showinfo("Éxito", "¡Gesto capturado correctamente!")
        else:
            messagebox.showwarning(
                "Captura Cancelada", "La captura del gesto fue cancelada."
            )
        self.open_capture_window()

    def stop_opencv_task(self):
        self.stop_event.set()
        self.after(200, self.cleanup_after_stop)

    def cleanup_after_stop(self):
        if self.opencv_controller:
            self.opencv_controller.release()
            self.opencv_controller = None
        if self.video_window:
            self.video_window.destroy()
            self.video_window = None
        self.update_button_states()

    def on_closing(self):
        # (Sin cambios)
        self.stop_opencv_task()
        self.after(250, self.destroy)

    # (El resto de las funciones auxiliares de la clase principal y las clases
    # de ventanas secundarias no necesitan cambios)
    def update_button_states(self):
        # ...
        config = load_config()
        model_exists = os.path.exists("models/gesture_prototypes.joblib")
        if not config:
            self.status_label.configure(text="Paso 1: Configura los jugadores.")
            self.btn_capture.configure(state="disabled")
            self.btn_train.configure(state="disabled")
            self.btn_play.configure(state="disabled")
            return
        capture_status = get_capture_status()
        all_captured = all(capture_status.values())
        self.btn_capture.configure(state="normal")
        if not all_captured:
            self.status_label.configure(
                text=f"Paso 2: Captura gestos para {len(capture_status)} jugadores."
            )
            self.btn_train.configure(state="disabled")
            self.btn_play.configure(state="disabled")
        else:
            self.status_label.configure(text="Paso 3: ¡Entrena el modelo!")
            self.btn_train.configure(state="normal")
            if model_exists:
                self.btn_play.configure(state="normal")
                self.status_label.configure(text="¡Listo para Jugar!")
            else:
                self.btn_play.configure(state="disabled")

    def open_setup_window(self):
        if self.video_window:
            return
        SetupWindow(self)

    def open_capture_window(self):
        if self.video_window:
            return
        CaptureWindow(self)

    def run_training(self):
        message, success = train_model()
        if success:
            messagebox.showinfo("Éxito", message)
        else:
            messagebox.showwarning("Error", message)
        self.update_button_states()

    def clean_and_reset(self):
        if self.video_window:
            return
        if messagebox.askyesno(
            "Confirmar Limpieza", "¿Borrar toda la configuración y datos?"
        ):
            cleanup_project_files()
            messagebox.showinfo("Limpieza", "Limpieza completada.")
            self.update_button_states()


class SetupWindow(ctk.CTkToplevel):
    # (El código es idéntico al de la versión anterior)
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title("Configurar Jugadores")
        self.geometry("350x400")
        self.transient(master)
        ctk.CTkLabel(self, text="Número de Jugadores:").pack(pady=(10, 0))
        self.num_players_combo = ctk.CTkComboBox(
            self,
            values=[str(i) for i in range(1, 5)],
            command=self.update_player_entries,
        )
        self.num_players_combo.pack()
        self.entries_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.entries_frame.pack(pady=10, fill="x", expand=True)
        self.player_entries = {}
        ctk.CTkButton(self, text="Guardar Configuración", command=self.save).pack(
            pady=10, side="bottom"
        )

    def update_player_entries(self, choice):
        [w.destroy() for w in self.entries_frame.winfo_children()]
        self.player_entries = {}
        [self.create_entry(i) for i in range(1, int(choice) + 1)]

    def create_entry(self, i):
        f = ctk.CTkFrame(self.entries_frame)
        ctk.CTkLabel(f, text=f"Tecla Jugador {i}:").pack(side="left", padx=5)
        e = ctk.CTkEntry(f, width=50)
        e.pack(side="left", padx=5)
        self.player_entries[str(i)] = e
        f.pack(pady=5)

    def save(self):
        try:
            num_p = int(self.num_players_combo.get())
            keys = {
                str(i): self.player_entries[str(i)].get().strip().lower()
                for i in range(1, num_p + 1)
            }
            [
                messagebox.showerror("Error", "Tecla inválida")
                for k in keys.values()
                if not k or len(k) > 1
            ]
            save_config(num_p, keys)
            messagebox.showinfo("Éxito", "Guardado")
            self.master.update_button_states()
            self.destroy()
        except (ValueError, KeyError):
            messagebox.showerror("Error", "Completa todos los campos")


class CaptureWindow(ctk.CTkToplevel):
    # (El código es idéntico al de la versión anterior)
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title("Captura de Gestos")
        self.geometry("350x300")
        self.transient(master)
        self.config = load_config()
        self.capture_status = get_capture_status()
        ctk.CTkLabel(
            self,
            text="Selecciona un jugador para capturar:",
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=10)
        for p_id, key in self.config["players"].items():
            self.create_capture_entry(p_id, key)

    def create_capture_entry(self, p_id, key):
        f = ctk.CTkFrame(self)
        status = "✅ Capturado" if self.capture_status.get(p_id) else "❌ Pendiente"
        color = "green" if self.capture_status.get(p_id) else "red"
        ctk.CTkLabel(
            f, text=f"Jugador {p_id} (Tecla: '{key}') - {status}", text_color=color
        ).pack(side="left", padx=10)
        btn = ctk.CTkButton(
            f, text="Capturar", command=lambda p=p_id, k=key: self.start_capture(p, k)
        )
        if self.capture_status.get(p_id):
            btn.configure(text="Re-capturar")
        btn.pack(side="right", padx=10)
        f.pack(pady=5, fill="x", padx=10)

    def start_capture(self, p_id, key):
        self.master.run_opencv_task("capture", p_id, key)
        self.destroy()
