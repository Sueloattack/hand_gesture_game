import numpy as np
import cv2


def normalize_landmarks(landmarks_mp):
    if landmarks_mp is None:
        return None

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_mp.landmark])

    if np.all(coords == 0):
        return None

    coords_translated = coords - coords[0]
    scaling_dist = np.linalg.norm(coords_translated[9])

    if scaling_dist < 1e-6:
        return None

    coords_normalized = coords_translated / scaling_dist
    return coords_normalized.flatten()


def get_player_zone(player, width, height, num_players):
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


def draw_zones(img, num_players, active_player=None):
    height, width, _ = img.shape
    for i in range(1, num_players + 1):
        x1, y1, x2, y2 = get_player_zone(i, width, height, num_players)
        color = (0, 255, 0)  # Verde por defecto
        thickness = 2
        if active_player is not None and i == active_player:
            color = (0, 255, 255)  # Amarillo para jugador activo
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
