import customtkinter as ctk
from tkinter import messagebox
import os
import threading
from PIL import Image
from customtkinter import CTkImage
import cv2

from app.core_logic import (
    save_config,
    load_config,
    get_capture_status,
    cleanup_project_files,
    train_model,
    OpenCVController,
    is_model_trained,
)

# Constantes para el centrado de la ventana de la cámara
CAM_WIDTH, CAM_HEIGHT = 640, 480
# Padding vertical para alojar el título, el label de muestras y la barra de progreso
VERTICAL_PADDING = 120


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
        self.frame_lock = threading.Lock()
        self.last_frame_data = {}
        self.setup_widgets()
        self.update_button_states()

    def setup_widgets(self):
        # (Sin cambios en esta función)
        ctk.CTkLabel(
            self,
            text="Control de Juegos por Gestos",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(pady=(20, 10))
        self.status_label = ctk.CTkLabel(self, text="Bienvenido.", wraplength=380)
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
        ctk.CTkButton(
            bottom_frame,
            text="Limpiar Todo",
            command=self.clean_and_reset,
            fg_color="#E74C3C",
            hover_color="#C0392B",
        ).pack(side="left", expand=True, padx=(0, 5))
        ctk.CTkButton(bottom_frame, text="Salir", command=self.on_closing).pack(
            side="right", expand=True, padx=(5, 0)
        )

    def run_opencv_task(self, mode, player_id=None, key=None):
        if self.video_window:
            return
        self.stop_event.clear()
        self.last_frame_data = {}

        try:
            self.opencv_controller = OpenCVController(self.update_gui_from_cv)
        except IOError as e:
            messagebox.showerror("Error de Cámara", str(e))
            return

        self.video_window = ctk.CTkToplevel(self)
        self.video_window.protocol("WM_DELETE_WINDOW", self.stop_opencv_task)

        # ***** CORRECCIÓN CLAVE: Centrar la ventana INMEDIATAMENTE *****
        self.center_toplevel_window(self.video_window)

        self.video_label = ctk.CTkLabel(self.video_window, text="Iniciando cámara...")
        self.video_label.pack(padx=10, pady=10)

        if mode == "capture":
            self.video_window.title(f"Capturando para Jugador {player_id}")
            self.capture_info_label = ctk.CTkLabel(self.video_window, text="")
            self.capture_info_label.pack(pady=(0, 5))
            self.progress_bar = ctk.CTkProgressBar(self.video_window)
            self.progress_bar.set(0)
            self.progress_bar.pack(pady=10, padx=10, fill="x")
            target = "run_capture"
            args = (player_id, key, self.stop_event)
            on_complete = self.handle_capture_completion
        else:
            self.video_window.title("Modo Juego")
            target = "run_play"
            args = (self.stop_event,)
            on_complete = None

        self.thread = threading.Thread(
            target=self.thread_wrapper,
            args=(getattr(self.opencv_controller, target), args, on_complete),
        )
        self.thread.start()
        self.render_loop()

    def center_toplevel_window(self, toplevel):
        # Lógica de centrado usando un tamaño predefinido para ser determinista
        window_width = (
            CAM_WIDTH + 40
        )  # Ancho de la cámara más un pequeño padding horizontal
        window_height = CAM_HEIGHT + VERTICAL_PADDING

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        toplevel.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def render_loop(self):
        if self.stop_event.is_set():
            return

        with self.frame_lock:
            frame_data = self.last_frame_data.copy()
        image = frame_data.get("image")

        if image is not None and self.video_window and self.video_label.winfo_exists():
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not hasattr(self.video_label, "_ctk_image_obj"):
                self.video_label._ctk_image_obj = CTkImage(
                    light_image=img_pil, size=img_pil.size
                )
            else:
                self.video_label._ctk_image_obj.configure(light_image=img_pil)
            self.video_label.configure(image=self.video_label._ctk_image_obj, text="")

            if (
                "progress" in frame_data
                and hasattr(self, "progress_bar")
                and self.progress_bar.winfo_exists()
            ):
                self.progress_bar.set(frame_data["progress"])
                self.capture_info_label.configure(
                    text=f"Muestras: {frame_data['count']} / {frame_data['total']}"
                )

        self.after(16, self.render_loop)

    def update_gui_from_cv(self, **kwargs):
        with self.frame_lock:
            self.last_frame_data = kwargs

    def stop_opencv_task(self):
        if not self.stop_event.is_set():
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

    def handle_capture_completion(self, success):
        self.stop_opencv_task()
        (
            messagebox.showinfo("Éxito", "¡Gesto capturado!")
            if success
            else messagebox.showwarning("Cancelado", "Captura cancelada o incompleta.")
        )
        self.open_capture_window()

    def thread_wrapper(self, target_func, args, on_complete):
        result = target_func(*args)
        if on_complete:
            self.after(0, lambda: on_complete(result))

    def on_closing(self):
        if self.video_window:
            self.stop_opencv_task()
        self.after(250, self.destroy)

    def update_button_states(self):
        config = load_config()
        model_exists = is_model_trained()
        self.btn_capture.configure(state="disabled")
        self.btn_train.configure(state="disabled")
        self.btn_play.configure(state="disabled")
        text = "Paso 1: Configura jugadores"
        if config:
            capture_status = get_capture_status()
            all_captured = all(capture_status.values())
            self.btn_capture.configure(state="normal")
            text = "Paso 2: Captura gestos"
            if all_captured:
                self.btn_train.configure(state="normal")
                text = "Paso 3: ¡Entrena el modelo!"
                if model_exists:
                    self.btn_play.configure(state="normal")
                    text = "¡Listo para Jugar!"
        self.status_label.configure(text=text)

    def open_setup_window(self):
        if not self.video_window:
            SetupWindow(self)

    def open_capture_window(self):
        if not self.video_window:
            CaptureWindow(self)

    def run_training(self):
        message, success = train_model()
        (messagebox.showinfo if success else messagebox.showwarning)(
            "Entrenamiento", message
        )
        self.update_button_states()

    def clean_and_reset(self):
        if not self.video_window and messagebox.askyesno(
            "Confirmar", "¿Borrar toda la configuración y datos?"
        ):
            messagebox.showinfo(
                "Limpieza", f"Se eliminaron {cleanup_project_files()} archivos."
            )
            self.update_button_states()


class SetupWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title("Configurar Jugadores")
        self.geometry("350x400")
        self.transient(master)
        ctk.CTkLabel(
            self,
            text="Por favor seleccione el número de jugadores",
            font=ctk.CTkFont(size=14),
        ).pack(pady=(20, 5))
        ctk.CTkLabel(self, text="Número de Jugadores:").pack(pady=(5, 0))
        self.num_players_combo = ctk.CTkComboBox(
            self,
            values=[str(i) for i in range(1, 5)],
            command=self.update_player_entries,
        )
        self.num_players_combo.pack()
        self.entries_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.entries_frame.pack(pady=10, fill="x", expand=True)
        self.player_entries = {}
        ctk.CTkButton(self, text="Guardar", command=self.save).pack(
            pady=20, side="bottom"
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
            if any(not key or len(key) != 1 for key in keys.values()):
                messagebox.showerror("Error", "Tecla debe ser un solo caracter.")
                return
            save_config(num_p, keys)
            messagebox.showinfo("Éxito", "Guardado")
            self.master.update_button_states()
            self.after(10, self.destroy)
        except (ValueError, KeyError):
            messagebox.showerror("Error", "Completa todos los campos")


class CaptureWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title("Captura de Gestos")
        self.geometry("420x300")
        self.transient(master)
        self.config = load_config()
        self.capture_status = get_capture_status()
        ctk.CTkLabel(
            self,
            text="Selecciona un jugador para capturar:",
            font=ctk.CTkFont(weight="bold"),
        ).pack(pady=10)
        if self.config:
            for p_id, key in self.config["players"].items():
                self.create_capture_entry(p_id, key)

    def create_capture_entry(self, p_id, key):
        frame = ctk.CTkFrame(self)
        status = "✅ Capturado" if self.capture_status.get(p_id) else "❌ Pendiente"
        color = (
            ("#2ECC71", "#27AE60")
            if self.capture_status.get(p_id)
            else ("#E74C3C", "#C0392B")
        )
        ctk.CTkLabel(frame, text=f"Jugador {p_id} (Tecla: '{key}') - {status}").pack(
            side="left", padx=10, pady=5
        )
        btn = ctk.CTkButton(
            frame,
            text="Re-capturar" if self.capture_status.get(p_id) else "Capturar",
            fg_color=color[0],
            hover_color=color[1],
            command=lambda p=p_id, k=key: self.start_capture(p, k),
        )
        btn.pack(side="right", padx=10, pady=5)
        frame.pack(pady=5, fill="x", padx=10)

    def start_capture(self, p_id, key):
        self.master.run_opencv_task("capture", p_id, key)
        self.destroy()
