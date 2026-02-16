import numpy as np
from core.snake_game import SnakeGame 
from RL.agent import DQNAgent


class SnakeEnv:
    """
    action: 0=left, 1=straight, 2=right
    """
    def __init__(self):
        self.game = SnakeGame()

    def reset(self):
        self.game.reset()
        return self.game.get_observation()

    def step(self, action):
        # -------- BEFORE moving (distance to food) --------
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        prev_dist = abs(head_x - food_x) + abs(head_y - food_y)

        # -------- take action --------
        result = self.game.step_action(action)

        # -------- AFTER moving (new distance) --------
        new_head_x, new_head_y = self.game.snake[0]
        new_dist = abs(new_head_x - food_x) + abs(new_head_y - food_y)

        # -------- base reward --------
        if result.done:
            reward = -100.0
        elif result.ate_food:
            reward = 10.0
        else:
            reward = -1.0

        # -------- distance shaping --------
        # Only apply when alive + not just ate food (keeps rewards stable)
        if (not result.done) and (not result.ate_food):
            reward += 0.2 if new_dist < prev_dist else -0.2

        next_state = self.game.get_observation()
        return next_state, reward, result.done, {"score": result.score}



def train(num_episodes=2000, save_every=100):
    env = SnakeEnv()
    agent = DQNAgent(state_size=11, action_size=3)

    best_score = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            score = info["score"]

        agent.end_episode()

        if score > best_score:
            best_score = score
            agent.save("Models/snake_dqn_best.pth")

        if episode % save_every == 0:
            agent.save("Models/snake_dqn.pth")

        print(
            f"Episode {episode}/{num_episodes} | Score: {score} | Best: {best_score} | Epsilon: {agent.epsilon:.3f}"
        )

    agent.save("Models/snake_dqn_final.pth")
    print("Training finished. Saved model.")


if __name__ == "__main__":
    train()
