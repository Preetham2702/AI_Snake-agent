# frontend/pygame_app.py
# Run from project root:
#   python -m frontend.pygame_app

import os
import glob
import shutil
import sys
import subprocess
import numpy as np
import pygame

from core.snake_game import SnakeGame, UP, DOWN, LEFT, RIGHT


# ---------------- UI THEME ----------------
CELL_SIZE = 25

BG = (245, 245, 245)
PANEL = (255, 255, 255)
OUTLINE = (210, 210, 210)
MUTED = (120, 120, 120)
TEXT = (20, 20, 20)

BAR = (230, 230, 230)
GRID = (235, 235, 235)
FOOD = (220, 50, 50)
HEAD = (30, 120, 30)
BODY = (60, 180, 60)

BTN = (35, 120, 220)
BTN_HOVER = (25, 95, 180)
BTN_TEXT = (255, 255, 255)

SELECTED_FILL = (235, 245, 235)
SELECTED_OUT = (30, 120, 30)

ERR = (180, 60, 60)

SPEED_LEVELS = [
    ("Low", 8),
    ("Medium", 12),
    ("Hard", 18),
    ("Extreme", 28),
]


# ---------------- AI OBSERVATION (feature vector) ----------------
# Feature vector (11):
# danger_ahead, danger_left, danger_right (3)
# food_up, food_down, food_left, food_right (4)
# dir_up, dir_down, dir_left, dir_right (4)
def build_features(game: SnakeGame) -> np.ndarray:
    snake = game.snake
    (hx, hy) = snake[0]
    dx, dy = game.direction
    (fx, fy) = game.food

    def collide(pos):
        x, y = pos
        if x < 0 or x >= game.width or y < 0 or y >= game.height:
            return 1.0
        if pos in snake:
            return 1.0
        return 0.0

    left_dir = (dy, -dx)
    right_dir = (-dy, dx)

    ahead = (hx + dx, hy + dy)
    left = (hx + left_dir[0], hy + left_dir[1])
    right = (hx + right_dir[0], hy + right_dir[1])

    danger_ahead = collide(ahead)
    danger_left = collide(left)
    danger_right = collide(right)

    food_up = 1.0 if fy < hy else 0.0
    food_down = 1.0 if fy > hy else 0.0
    food_left = 1.0 if fx < hx else 0.0
    food_right = 1.0 if fx > hx else 0.0

    dir_up = 1.0 if (dx, dy) == (0, -1) else 0.0
    dir_down = 1.0 if (dx, dy) == (0, 1) else 0.0
    dir_left = 1.0 if (dx, dy) == (-1, 0) else 0.0
    dir_right = 1.0 if (dx, dy) == (1, 0) else 0.0

    return np.array(
        [
            danger_ahead,
            danger_left,
            danger_right,
            food_up,
            food_down,
            food_left,
            food_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
        ],
        dtype=np.float32,
    )


# ---------------- Helpers ----------------
def point_in_rect(pos, rect: pygame.Rect) -> bool:
    return rect.collidepoint(pos[0], pos[1])


def draw_center_text(screen, font, msg, rect: pygame.Rect, color):
    surf = font.render(msg, True, color)
    screen.blit(
        surf,
        (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2),
    )


def draw_text(screen, font, msg, x, y, color=TEXT):
    screen.blit(font.render(msg, True, color), (x, y))


def pick_model_file():
    """
    macOS-safe file picker:
    Runs tkinter filedialog in a separate process to avoid SDL/Pygame crash.
    Returns selected path or "".
    """
    code = r"""
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

path = filedialog.askopenfilename(
    title="Select Stable-Baselines3 model (.zip)",
    filetypes=[("SB3 Model", "*.zip")]
)

try:
    root.destroy()
except Exception:
    pass

print(path if path else "")
"""
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        return (res.stdout or "").strip()
    except Exception:
        return ""


def load_models_list():
    paths = glob.glob(os.path.join("models", "*.zip"))
    paths.sort()
    return paths


def try_load_sb3_model(path: str):
    # Lazy import so HUMAN mode works even without SB3 installed
    from stable_baselines3 import DQN, PPO

    for cls in (DQN, PPO):
        try:
            return cls.load(path)
        except Exception:
            pass
    raise RuntimeError(
        "Could not load model. Ensure it's a Stable-Baselines3 DQN/PPO .zip model."
    )


# ---------------- Main App ----------------
def main():
    pygame.init()

    grid_w, grid_h = 20, 20
    game = SnakeGame(grid_w, grid_h)

    width_px = grid_w * CELL_SIZE
    height_px = grid_h * CELL_SIZE + 90
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("AI Snake Game (Human / AI)")

    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("Arial", 40, bold=True)
    font = pygame.font.SysFont("Arial", 22)
    small = pygame.font.SysFont("Arial", 18)
    tiny = pygame.font.SysFont("Arial", 16)

    # modes:
    # choose_mode -> choose_settings_human -> game_human
    # choose_settings_ai -> game_ai
    mode = "choose_mode"

    selected_speed_idx = 1  # Medium
    fps = SPEED_LEVELS[selected_speed_idx][1]

    current_dir = RIGHT
    paused = False

    models = load_models_list()
    selected_model_idx = 0
    ai_model = None
    ai_error = ""

    # ----- layout -----
    center_x = width_px // 2
    panel = pygame.Rect(center_x - 280, 70, 560, 420)

    # Choose mode buttons
    btn_w = 240
    btn_h = 50
    gap_y = 18
    human_btn = pygame.Rect(center_x - btn_w // 2, panel.y + 150, btn_w, btn_h)
    ai_btn = pygame.Rect(center_x - btn_w // 2, human_btn.y + btn_h + gap_y, btn_w, btn_h)

    # Settings buttons (small + margins)
    back_btn = pygame.Rect(panel.x + 30, panel.bottom - 52, 100, 36)
    start_btn = pygame.Rect(panel.right - 130, panel.bottom - 52, 100, 36)

    # Upload button centered
    upload_btn = pygame.Rect(center_x - 140, panel.y + 95, 280, 42)

    # speed buttons 2x2
    speed_btns = []
    sb_w, sb_h, gap = 200, 46, 14
    start_x = center_x - (2 * sb_w + gap) // 2
    row1_y = panel.y + 170
    row2_y = row1_y + sb_h + gap
    speed_btns.append((0, pygame.Rect(start_x, row1_y, sb_w, sb_h)))
    speed_btns.append((1, pygame.Rect(start_x + sb_w + gap, row1_y, sb_w, sb_h)))
    speed_btns.append((2, pygame.Rect(start_x, row2_y, sb_w, sb_h)))
    speed_btns.append((3, pygame.Rect(start_x + sb_w + gap, row2_y, sb_w, sb_h)))

    def reset_game():
        nonlocal current_dir, paused
        game.reset()
        current_dir = RIGHT
        paused = False

    def start_human():
        nonlocal mode
        reset_game()
        mode = "game_human"

    def start_ai():
        nonlocal mode, ai_model, ai_error, models
        reset_game()
        ai_error = ""
        models = load_models_list()
        if not models:
            ai_error = "No models found."
            return
        try:
            ai_model = try_load_sb3_model(models[selected_model_idx])
            mode = "game_ai"
        except Exception as e:
            ai_error = str(e)

    running = True
    while running:
        mouse = pygame.mouse.get_pos()
        clock.tick(fps if mode.startswith("game") else 60)

        # -------- events --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ----- choose mode -----
            if mode == "choose_mode":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if point_in_rect(mouse, human_btn):
                        mode = "choose_settings_human"
                    elif point_in_rect(mouse, ai_btn):
                        mode = "choose_settings_ai"

            # ----- human settings -----
            elif mode == "choose_settings_human":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for idx, rect in speed_btns:
                        if point_in_rect(mouse, rect):
                            selected_speed_idx = idx
                            fps = SPEED_LEVELS[selected_speed_idx][1]
                    if point_in_rect(mouse, back_btn):
                        mode = "choose_mode"
                    elif point_in_rect(mouse, start_btn):
                        start_human()

            # ----- ai settings -----
            elif mode == "choose_settings_ai":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Upload
                    if point_in_rect(mouse, upload_btn):
                        picked = pick_model_file()
                        if picked:
                            os.makedirs("models", exist_ok=True)
                            dest = os.path.join("models", os.path.basename(picked))
                            try:
                                shutil.copy(picked, dest)
                                models = load_models_list()
                                selected_model_idx = models.index(dest) if dest in models else 0
                                ai_error = ""
                            except Exception:
                                ai_error = "Upload failed"
                        else:
                            ai_error = "No file selected"

                    # Speed
                    for idx, rect in speed_btns:
                        if point_in_rect(mouse, rect):
                            selected_speed_idx = idx
                            fps = SPEED_LEVELS[selected_speed_idx][1]

                    # Model list click
                    list_x = panel.x + 40
                    list_y = upload_btn.bottom + 65
                    row_h = 28
                    for i, p in enumerate(models[:10]):
                        row_rect = pygame.Rect(list_x, list_y + i * row_h, panel.width - 80, row_h)
                        if point_in_rect(mouse, row_rect):
                            selected_model_idx = i

                    # Back / Start
                    if point_in_rect(mouse, back_btn):
                        mode = "choose_mode"
                    elif point_in_rect(mouse, start_btn):
                        start_ai()

                # Drag & drop file
                if event.type == pygame.DROPFILE:
                    dropped = event.file
                    if dropped.lower().endswith(".zip"):
                        os.makedirs("models", exist_ok=True)
                        dest = os.path.join("models", os.path.basename(dropped))
                        try:
                            shutil.copy(dropped, dest)
                            models = load_models_list()
                            selected_model_idx = models.index(dest) if dest in models else 0
                            ai_error = ""
                        except Exception:
                            ai_error = "Upload failed"
                    else:
                        ai_error = "Drop a .zip file"

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_model_idx = max(0, selected_model_idx - 1)
                    elif event.key == pygame.K_DOWN:
                        selected_model_idx = min(max(0, len(models) - 1), selected_model_idx + 1)

            # ----- human game -----
            elif mode == "game_human":
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_w, pygame.K_UP):
                        current_dir = UP
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        current_dir = DOWN
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        current_dir = LEFT
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        current_dir = RIGHT
                    elif event.key == pygame.K_r:
                        reset_game()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        mode = "choose_mode"

            # ----- ai game -----
            elif mode == "game_ai":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        reset_game()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        mode = "choose_mode"

        # -------- update game --------
        st = None
        if mode == "game_human":
            st = game.step(current_dir) if (not paused and not game.done) else game.get_state()

        elif mode == "game_ai":
            if not paused and not game.done:
                obs = build_features(game)
                action, _ = ai_model.predict(obs, deterministic=True)
                st = game.step_action(int(action))
            else:
                st = game.get_state()

        # -------- draw --------
        screen.fill(BG)

        # ----- choose mode -----
        if mode == "choose_mode":
            pygame.draw.rect(screen, PANEL, panel, border_radius=16)
            pygame.draw.rect(screen, OUTLINE, panel, width=2, border_radius=16)

            draw_text(screen, title_font, "AI SNAKE GAME", panel.x + 120, panel.y + 45)
            draw_text(screen, font, "Choose Mode", panel.x + 210, panel.y + 110, color=MUTED)

            hover = point_in_rect(mouse, human_btn)
            pygame.draw.rect(screen, BTN_HOVER if hover else BTN, human_btn, border_radius=14)
            draw_center_text(screen, font, "HUMAN", human_btn, BTN_TEXT)

            hover = point_in_rect(mouse, ai_btn)
            pygame.draw.rect(screen, BTN_HOVER if hover else BTN, ai_btn, border_radius=14)
            draw_center_text(screen, font, "AI", ai_btn, BTN_TEXT)

            desc_y = ai_btn.bottom + 28
            t1 = small.render("Human: play with keyboard", True, MUTED)
            screen.blit(t1, (center_x - t1.get_width() // 2, desc_y))
            t2 = small.render("AI: loads a trained model from /models", True, MUTED)
            screen.blit(t2, (center_x - t2.get_width() // 2, desc_y + 24))

        # ----- settings screens -----
        elif mode in ("choose_settings_human", "choose_settings_ai"):
            pygame.draw.rect(screen, PANEL, panel, border_radius=16)
            pygame.draw.rect(screen, OUTLINE, panel, width=2, border_radius=16)

            title = "Human Settings" if mode == "choose_settings_human" else "AI Settings"
            draw_text(screen, title_font, title, panel.x + 165, panel.y + 35)

            # Speed buttons (NO label)
            for idx, rect in speed_btns:
                label, _ = SPEED_LEVELS[idx]
                is_sel = idx == selected_speed_idx
                pygame.draw.rect(screen, SELECTED_FILL if is_sel else (245, 245, 245), rect, border_radius=12)
                pygame.draw.rect(screen, SELECTED_OUT if is_sel else OUTLINE, rect, width=2, border_radius=12)
                draw_center_text(screen, font, label, rect, (20, 90, 20) if is_sel else TEXT)

            if mode == "choose_settings_ai":
                # Upload centered
                pygame.draw.rect(screen, (245, 245, 245), upload_btn, border_radius=12)
                pygame.draw.rect(screen, OUTLINE, upload_btn, width=2, border_radius=12)
                upload_txt = font.render("Upload Model", True, TEXT)
                screen.blit(upload_txt, (upload_btn.centerx - upload_txt.get_width() // 2,
                                        upload_btn.centery - upload_txt.get_height() // 2))

                # Hint + small inline error beside hint
                hint = tiny.render("(or drag & drop .zip)", True, MUTED)
                hint_x = center_x - hint.get_width() // 2
                hint_y = upload_btn.bottom + 6
                screen.blit(hint, (hint_x, hint_y))

                if ai_error:
                    err = tiny.render(ai_error, True, ERR)
                    screen.blit(err, (hint_x + hint.get_width() + 12, hint_y))

                # Model list (NO "Models" label)
                list_x = panel.x + 40
                list_y = upload_btn.bottom + 65
                row_h = 28

                if models:
                    for i, p in enumerate(models[:10]):
                        name = os.path.basename(p)
                        is_sel = i == selected_model_idx
                        row_rect = pygame.Rect(list_x, list_y + i * row_h, panel.width - 80, row_h)
                        pygame.draw.rect(screen, (235, 245, 235) if is_sel else (250, 250, 250), row_rect, border_radius=6)
                        pygame.draw.rect(screen, SELECTED_OUT if is_sel else OUTLINE, row_rect, width=1, border_radius=6)
                        draw_text(screen, tiny, name, row_rect.x + 10, row_rect.y + 6, color=TEXT)

            # Back / Start small + margins
            pygame.draw.rect(screen, (245, 245, 245), back_btn, border_radius=10)
            pygame.draw.rect(screen, OUTLINE, back_btn, width=2, border_radius=10)
            draw_center_text(screen, small, "Back", back_btn, TEXT)

            hover = point_in_rect(mouse, start_btn)
            pygame.draw.rect(screen, BTN_HOVER if hover else BTN, start_btn, border_radius=10)
            draw_center_text(screen, small, "Start", start_btn, BTN_TEXT)

            draw_text(
                screen,
                tiny,
                f"Selected Speed: {SPEED_LEVELS[selected_speed_idx][0]} ({fps} FPS)",
                center_x - 110,
                panel.y + panel.height - 95,
                color=MUTED,
            )

        # ----- game screens -----
        elif mode in ("game_human", "game_ai"):
            pygame.draw.rect(screen, BAR, (0, 0, width_px, 90))
            draw_text(screen, font,
                      f"Score: {st.score}   Mode: {'HUMAN' if mode=='game_human' else 'AI'}   Speed: {SPEED_LEVELS[selected_speed_idx][0]}",
                      10, 12)
            draw_text(screen, small, "[R] Restart   [Space] Pause   [Esc] Menu", 10, 45, color=MUTED)

            offset_y = 90

            for x in range(grid_w + 1):
                pygame.draw.line(screen, GRID, (x * CELL_SIZE, offset_y),
                                 (x * CELL_SIZE, offset_y + grid_h * CELL_SIZE))
            for y in range(grid_h + 1):
                pygame.draw.line(screen, GRID, (0, offset_y + y * CELL_SIZE),
                                 (grid_w * CELL_SIZE, offset_y + y * CELL_SIZE))

            if st.food != (-1, -1):
                fx, fy = st.food
                food_rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE + offset_y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, FOOD, food_rect, border_radius=6)

            for i, (x, y) in enumerate(st.snake):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + offset_y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, HEAD if i == 0 else BODY, rect, border_radius=6)

            if paused and not game.done:
                draw_text(screen, font, "PAUSED", width_px // 2 - 55, 20)

            if game.done:
                overlay = pygame.Surface((width_px, grid_h * CELL_SIZE), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                screen.blit(overlay, (0, offset_y))
                draw_text(screen, font, "GAME OVER", width_px // 2 - 75,
                          offset_y + (grid_h * CELL_SIZE) // 2 - 30, color=(255, 255, 255))
                draw_text(screen, small, "Press R to restart or Esc for menu",
                          width_px // 2 - 145, offset_y + (grid_h * CELL_SIZE) // 2 + 5, color=(255, 255, 255))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
