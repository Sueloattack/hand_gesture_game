import json
import os
import glob
import numpy as np
import pandas as pd
import joblib
import cv2
import mediapipe as mp
from pynput.keyboard import Controller

from app.utils import normalize_landmarks, get_player_zone, draw_zones

CONFIG_FILE = "config.json"
DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_prototypes.joblib")


def save_config(num_players, players_keys):
    config = {"num_players": num_players, "players": players_keys}
    f = open(CONFIG_FILE, "w")
    json.dump(config, f, indent=4)
    f.close()
    return True


def load_config():
    return json.load(open(CONFIG_FILE, "r")) if os.path.exists(CONFIG_FILE) else None


def get_capture_status():
    config = load_config()
    return (
        {
            p: os.path.exists(os.path.join(DATA_DIR, f"{k}.csv"))
            for p, k in config["players"].items()
        }
        if config
        else {}
    )


# --- NUEVA FUNCIÓN ---
def is_model_trained():
    """Revisa si el archivo del modelo de gestos ha sido entrenado y existe."""
    return os.path.exists(MODEL_PATH)


def cleanup_project_files():
    files_to_check = (
        [CONFIG_FILE]
        + glob.glob(os.path.join(DATA_DIR, "*.csv"))
        + glob.glob(os.path.join(MODELS_DIR, "*.joblib"))
    )
    files_deleted_count = 0
    for f in files_to_check:
        if os.path.exists(f):
            os.remove(f)
            files_deleted_count += 1
    return files_deleted_count


def train_model():
    if not any(f.endswith(".csv") for f in os.listdir(DATA_DIR)):
        return "No hay datos para entrenar.", False
    prototypes, thresholds = {}, {}
    os.makedirs(MODELS_DIR, exist_ok=True)
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            label = os.path.splitext(file)[0]
            X = pd.read_csv(os.path.join(DATA_DIR, file), header=None).values
            if X.shape[0] > 0:
                proto = X.mean(axis=0)
                dists = np.linalg.norm(X - proto, axis=1)
                thr = dists.max() * 1.2
                prototypes[label], thresholds[label] = proto, thr
    if not prototypes:
        return "No se generaron prototipos válidos.", False
    joblib.dump({"prototypes": prototypes, "thresholds": thresholds}, MODEL_PATH)
    return f"Modelo guardado con {len(prototypes)} gestos.", True


class OpenCVController:
    # (El resto de esta clase no cambia en absoluto)
    def __init__(self, update_callback):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("No se puede abrir la webcam")
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.kb = Controller()
        self.update_callback = update_callback

    def run_capture(self, player_id, key, stop_event):
        SAMPLES = 300
        gesture_data = []
        with self.mp_hands.Hands(
            min_detection_confidence=0.7, max_num_hands=1
        ) as hands:
            while len(gesture_data) < SAMPLES and not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                img = cv2.flip(frame, 1)
                h, w, _ = img.shape
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                config = load_config()
                draw_zones(img, config["num_players"], active_player=int(player_id))
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        cx = np.mean([p.x for p in lm.landmark]) * w
                        cy = np.mean([p.y for p in lm.landmark]) * h
                        x1, y1, x2, y2 = get_player_zone(
                            int(player_id), w, h, config["num_players"]
                        )
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            normalized_data = normalize_landmarks(lm)
                            if normalized_data is not None:
                                gesture_data.append(normalized_data)
                            self.mp_draw.draw_landmarks(
                                img, lm, self.mp_hands.HAND_CONNECTIONS
                            )
                self.update_callback(
                    image=img,
                    progress=len(gesture_data) / SAMPLES,
                    count=len(gesture_data),
                    total=SAMPLES,
                )
        success = len(gesture_data) >= SAMPLES
        if success:
            np.savetxt(
                os.path.join(DATA_DIR, f"{key}.csv"), gesture_data, delimiter=","
            )
        return success

    def run_play(self, stop_event):
        config = load_config()
        model = joblib.load(MODEL_PATH)
        protos, thresholds = model["prototypes"], model["thresholds"]
        keys_pressed = {key: False for key in config["players"].values()}
        with self.mp_hands.Hands(
            min_detection_confidence=0.7, max_num_hands=config["num_players"]
        ) as hands:
            while not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break
                img = cv2.flip(frame, 1)
                h, w, _ = img.shape
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw_zones(img, config["num_players"])
                current_gestures = {key: False for key in config["players"].values()}
                if results.multi_hand_landmarks:
                    for lm in results.multi_hand_landmarks:
                        normalized_data = normalize_landmarks(lm)
                        if normalized_data is None:
                            continue
                        cx = np.mean([p.x for p in lm.landmark]) * w
                        cy = np.mean([p.y for p in lm.landmark]) * h
                        for i, key in config["players"].items():
                            x1, y1, x2, y2 = get_player_zone(
                                int(i), w, h, config["num_players"]
                            )
                            if x1 <= cx <= x2 and y1 <= cy <= y2 and key in protos:
                                d = np.linalg.norm(normalized_data - protos[key])
                                color = (0, 0, 255)
                                if d < thresholds[key]:
                                    current_gestures[key] = True
                                    color = (0, 255, 0)
                                ds = self.mp_draw.DrawingSpec(color=color, thickness=2)
                                self.mp_draw.draw_landmarks(
                                    img, lm, self.mp_hands.HAND_CONNECTIONS, ds, ds
                                )
                                break
                for key, active in current_gestures.items():
                    if active and not keys_pressed[key]:
                        self.kb.press(key)
                        keys_pressed[key] = True
                    elif not active and keys_pressed[key]:
                        self.kb.release(key)
                        keys_pressed[key] = False
                self.update_callback(image=img)

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
