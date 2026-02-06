import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    """
    Pure game logic: reset() and step() for Human,
    plus step_action() for AI (left/straight/right).
    """
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
        if new_head in self.snake:
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
