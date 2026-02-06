import pygame
from core.snake_game import SnakeGame, UP, DOWN, LEFT, RIGHT

CELL_SIZE = 25

# Colors
BG = (245, 245, 245)
PANEL = (255, 255, 255)
BAR = (230, 230, 230)
GRID = (235, 235, 235)
FOOD = (220, 50, 50)
HEAD = (30, 120, 30)
BODY = (60, 180, 60)
TEXT = (20, 20, 20)
MUTED = (110, 110, 110)
BTN = (35, 120, 220)
BTN_HOVER = (25, 95, 180)
BTN_TEXT = (255, 255, 255)
OUTLINE = (210, 210, 210)
SELECTED = (30, 120, 30)

SPEED_LEVELS = [
    ("Low", 8),
    ("Medium", 12),
    ("Hard", 18),
    ("Extreme", 28),
]

def draw_text(screen, font, msg, x, y, color=TEXT):
    img = font.render(msg, True, color)
    screen.blit(img, (x, y))

def point_in_rect(pos, rect: pygame.Rect) -> bool:
    return rect.collidepoint(pos[0], pos[1])

def main():
    pygame.init()

    grid_w, grid_h = 20, 20
    game = SnakeGame(grid_w, grid_h)

    width_px = grid_w * CELL_SIZE
    height_px = grid_h * CELL_SIZE + 70  # top bar space
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Snake (Menu + Speed Levels)")

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 22)
    title_font = pygame.font.SysFont("Arial", 40, bold=True)
    small_font = pygame.font.SysFont("Arial", 18)

    current_dir = RIGHT
    paused = False

    # App states: "menu" or "game"
    state_mode = "menu"
    selected_speed_idx = 1  # default Medium
    fps = SPEED_LEVELS[selected_speed_idx][1]

    def start_game():
        nonlocal state_mode, paused, current_dir
        game.reset()
        current_dir = RIGHT
        paused = False
        state_mode = "game"

    def back_to_menu():
        nonlocal state_mode, paused
        paused = False
        state_mode = "menu"

    # Precompute menu layout
    menu_center_x = width_px // 2
    menu_top = 110

    # Buttons (rects updated each frame in case window changes - but here fixed)
    start_btn = pygame.Rect(menu_center_x - 140, menu_top + 230, 280, 55)

    speed_btns = []
    btn_w, btn_h, gap = 170, 46, 14
    start_x = menu_center_x - (2 * btn_w + gap) // 2
    row1_y = menu_top + 110
    row2_y = row1_y + btn_h + gap

    # 4 speed buttons in 2x2
    speed_btns.append((0, pygame.Rect(start_x, row1_y, btn_w, btn_h)))
    speed_btns.append((1, pygame.Rect(start_x + btn_w + gap, row1_y, btn_w, btn_h)))
    speed_btns.append((2, pygame.Rect(start_x, row2_y, btn_w, btn_h)))
    speed_btns.append((3, pygame.Rect(start_x + btn_w + gap, row2_y, btn_w, btn_h)))

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # -------- events --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if state_mode == "menu":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # speed selection
                    for idx, rect in speed_btns:
                        if point_in_rect(mouse_pos, rect):
                            selected_speed_idx = idx
                            fps = SPEED_LEVELS[selected_speed_idx][1]

                    # start
                    if point_in_rect(mouse_pos, start_btn):
                        start_game()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        start_game()

            else:  # game mode
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
                        start_game()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        back_to_menu()

        # -------- update --------
        clock.tick(fps if state_mode == "game" else 60)

        # -------- draw --------
        screen.fill(BG)

        if state_mode == "menu":
            # Card/panel
            panel = pygame.Rect(menu_center_x - 260, 70, 520, 360)
            pygame.draw.rect(screen, PANEL, panel, border_radius=16)
            pygame.draw.rect(screen, OUTLINE, panel, width=2, border_radius=16)

            draw_text(screen, title_font, "AI Snake Game", panel.x + 140, panel.y + 30)
            draw_text(screen, font, "Choose Speed Level", panel.x + 160, panel.y + 90, color=MUTED)

            # Speed buttons
            for idx, rect in speed_btns:
                label, _ = SPEED_LEVELS[idx]
                is_selected = (idx == selected_speed_idx)

                pygame.draw.rect(
                    screen,
                    (235, 245, 235) if is_selected else (245, 245, 245),
                    rect,
                    border_radius=12
                )
                pygame.draw.rect(
                    screen,
                    SELECTED if is_selected else OUTLINE,
                    rect,
                    width=2,
                    border_radius=12
                )

                # Center text
                text_color = (20, 90, 20) if is_selected else TEXT
                txt = font.render(label, True, text_color)
                screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

            # Start button (hover effect)
            hover = point_in_rect(mouse_pos, start_btn)
            pygame.draw.rect(screen, BTN_HOVER if hover else BTN, start_btn, border_radius=14)
            start_txt = font.render("Start", True, BTN_TEXT)
            screen.blit(start_txt, (start_btn.centerx - start_txt.get_width() // 2,
                                   start_btn.centery - start_txt.get_height() // 2))

            draw_text(screen, small_font, f"Selected: {SPEED_LEVELS[selected_speed_idx][0]} ({fps} FPS)",
                      panel.x + 150, panel.y + 330, color=MUTED)

        else:
            # ---- game draw ----
            # Top bar
            pygame.draw.rect(screen, BAR, (0, 0, width_px, 70))
            st = game.get_state()
            draw_text(screen, font, f"Score: {st.score}   Speed: {SPEED_LEVELS[selected_speed_idx][0]}",
                      10, 12)
            draw_text(screen, small_font, "[R] Restart   [Space] Pause   [Esc] Menu", 10, 42, color=MUTED)

            offset_y = 70

            # Step the game
            if not paused and not game.done:
                st = game.step(current_dir)
            else:
                st = game.get_state()

            # Grid lines
            for x in range(grid_w + 1):
                pygame.draw.line(screen, GRID, (x * CELL_SIZE, offset_y),
                                 (x * CELL_SIZE, offset_y + grid_h * CELL_SIZE))
            for y in range(grid_h + 1):
                pygame.draw.line(screen, GRID, (0, offset_y + y * CELL_SIZE),
                                 (grid_w * CELL_SIZE, offset_y + y * CELL_SIZE))

            # Food
            if st.food != (-1, -1):
                fx, fy = st.food
                food_rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE + offset_y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, FOOD, food_rect, border_radius=6)

            # Snake
            for i, (x, y) in enumerate(st.snake):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + offset_y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, HEAD if i == 0 else BODY, rect, border_radius=6)

            # Overlays
            if paused and not game.done:
                draw_text(screen, font, "PAUSED", width_px // 2 - 45, 20, color=TEXT)

            if game.done:
                overlay = pygame.Surface((width_px, grid_h * CELL_SIZE), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                screen.blit(overlay, (0, offset_y))
                draw_text(screen, font, "GAME OVER", width_px // 2 - 70, offset_y + (grid_h * CELL_SIZE)//2 - 30, color=(255, 255, 255))
                draw_text(screen, small_font, "Press R to restart or Esc for menu",
                          width_px // 2 - 140, offset_y + (grid_h * CELL_SIZE)//2 + 5, color=(255, 255, 255))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
