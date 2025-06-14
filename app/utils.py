import numpy as np
import cv2


def get_player_zone(player, width, height, num_players):
    """Calcula las coordenadas de la zona de un jugador en la pantalla."""
    if num_players == 1:
        return 0, 0, width, height
    elif num_players == 2:
        zone_width = width // 2
        x1 = (player - 1) * zone_width
        return x1, 0, x1 + zone_width, height
    elif num_players == 3:
        if player in [1, 2]:
            zone_width = width // 2
            zone_height = height // 2
            x1 = (player - 1) * zone_width
            return x1, 0, x1 + zone_width, zone_height
        else:
            return 0, height // 2, width, height
    elif num_players == 4:
        zone_width = width // 2
        zone_height = height // 2
        x1 = ((player - 1) % 2) * zone_width
        y1 = ((player - 1) // 2) * zone_height
        return x1, y1, x1 + zone_width, y1 + zone_height
    return None


def draw_zones(img, num_players, active_player=None):
    """Dibuja las zonas de los jugadores en la imagen."""
    height, width, _ = img.shape
    for i in range(1, num_players + 1):
        x1, y1, x2, y2 = get_player_zone(i, width, height, num_players)
        color = (0, 255, 0)
        thickness = 2
        if active_player is not None and i == active_player:
            color = (0, 255, 255)
            thickness = 4
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img,
            f"Jugador {i}",
            (x1 + 10, y1 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )


def normalize_landmarks(hand_landmarks):
    """
    Normaliza los 21x3 landmarks para que sean invariantes a la escala,
    posición y rotación de la mano. Devuelve un vector de 63 elementos.
    """
    if hand_landmarks is None:
        return None

    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # 1. Centrar en el origen (usando la muñeca como punto de referencia)
    coords_translated = coords - coords[0]

    # 2. Normalizar la escala
    scaling_dist = np.linalg.norm(coords_translated[9] - coords_translated[0])
    if scaling_dist < 1e-6:
        return None

    coords_normalized = coords_translated / scaling_dist

    # 3. Aplanar el array a un vector de (63,)
    return coords_normalized.flatten()
