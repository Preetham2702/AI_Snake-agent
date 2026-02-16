import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

Pos = Tuple[int, int]
Dir = Tuple[int, int]

UP: Dir = (0, -1)
DOWN: Dir = (0, 1)
LEFT: Dir = (-1, 0)
RIGHT: Dir = (1, 0)

@dataclass
class StepResult:
    snake: List[Pos]
    food: Pos
    direction: Dir
    score: int
    done: bool
    ate_food: bool

class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20, init_length: int = 3):
        self.width = width
        self.height = height
        self.init_length = max(2, init_length)
        self.reset()

    def reset(self) -> StepResult:
        cx, cy = self.width // 2, self.height // 2
        self.direction: Dir = RIGHT
        self.snake: List[Pos] = [(cx - i, cy) for i in range(self.init_length)]
        self.score = 0
        self.done = False
        self._spawn_food()
        return self._result(ate_food=False)

    def _spawn_food(self) -> None:
        empty = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.snake]
        self.food = random.choice(empty) if empty else (-1, -1)

    @staticmethod
    def _is_opposite(a: Dir, b: Dir) -> bool:
        return a[0] == -b[0] and a[1] == -b[1]

    @staticmethod
    def _turn_left(d: Dir) -> Dir:
        dx, dy = d
        return (dy, -dx)

    @staticmethod
    def _turn_right(d: Dir) -> Dir:
        dx, dy = d
        return (-dy, dx)

    def get_state(self) -> StepResult:
        return self._result(ate_food=False)

    # ---------- Human step (absolute direction) ----------
    def step(self, new_direction: Optional[Dir] = None) -> StepResult:
        if self.done:
            return self._result(ate_food=False)

        if new_direction is not None and not self._is_opposite(new_direction, self.direction):
            self.direction = new_direction

        return self._advance()

    # ---------- AI step (relative action) ----------
    def step_action(self, action: int) -> StepResult:
        """
        action: 0=left, 1=straight, 2=right
        """
        if self.done:
            return self._result(ate_food=False)

        if action == 0:
            self.direction = self._turn_left(self.direction)
        elif action == 2:
            self.direction = self._turn_right(self.direction)
        # action==1 -> straight (no change)

        return self._advance()

    def _advance(self) -> StepResult:
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head: Pos = (hx + dx, hy + dy)

        # Wall collision
        if not (0 <= new_head[0] < self.width and 0 <= new_head[1] < self.height):
            self.done = True
            return self._result(ate_food=False)

        # Self collision
        if new_head in self.snake[:-1]:
            self.done = True
            return self._result(ate_food=False)

        self.snake.insert(0, new_head)

        ate_food = (new_head == self.food)
        if ate_food:
            self.score += 1
            self._spawn_food()
        else:
            self.snake.pop()

        if self.food == (-1, -1):
            self.done = True

        return self._result(ate_food=ate_food)

    def _result(self, ate_food: bool) -> StepResult:
        return StepResult(
            snake=list(self.snake),
            food=self.food,
            direction=self.direction,
            score=self.score,
            done=self.done,
            ate_food=ate_food,
        )

    def get_observation(self) -> np.ndarray:
        head = self.snake[0]

        def will_collide(pos):
            x, y = pos
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                return True
            # tail moves away if not eating
            if pos in self.snake[:-1]:
                return True
            return False

        straight = self.direction
        left = self._turn_left(self.direction)
        right = self._turn_right(self.direction)

        def next_pos(d):
            return (head[0] + d[0], head[1] + d[1])

        danger_straight = 1 if will_collide(next_pos(straight)) else 0
        danger_left = 1 if will_collide(next_pos(left)) else 0
        danger_right = 1 if will_collide(next_pos(right)) else 0

        food_left = 1 if self.food[0] < head[0] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_up = 1 if self.food[1] < head[1] else 0
        food_down = 1 if self.food[1] > head[1] else 0

        moving_left = 1 if self.direction == LEFT else 0
        moving_right = 1 if self.direction == RIGHT else 0
        moving_up = 1 if self.direction == UP else 0
        moving_down = 1 if self.direction == DOWN else 0

        return np.array([
            danger_straight, danger_left, danger_right,
            food_left, food_right, food_up, food_down,
            moving_left, moving_right, moving_up, moving_down
        ], dtype=np.float32)
