import time

from maze import Maze
from q_learning import QLearning

import time   # 追加
from q_learning import QLearning   # 追加



def main():
    
    ## Q学習で必要な条件
    
    # 何回ゴールするか
    EPISODE_MAX = 50
    # ゴールまでの打ち切りステップ数
    STEP_MAX = 3000
    # 学習率
    # LEARNING_RATE = 0.2
    LEARNING_RATE = 0.3
    # 割引率
    DISCOUNT_RATE = 0.95
    # 描画スピード
    SLEEP_TIME = 0.015
    
    # MazeとQLearningを用いた処理を記述
    
    maze = Maze()
    q_learn = QLearning(maze)
    
    for episode in range(EPISODE_MAX):
        step = 0
        q_learn.from_start()
        random_rate = 0.01 + 0.9 / (1 + episode)
        while not maze.is_goal() and step < STEP_MAX:
            q_learn.step(LEARNING_RATE, DISCOUNT_RATE, random_rate)
            maze.draw()
            step += 1
            time.sleep(SLEEP_TIME)
        print(f"episode : {episode} step : {step} ")
        print(..., end="", flush=True)   # 行末までをクリア

    for episode in range(EPISODE_MAX):
        step = 0
        q_learn.from_start()
        # random_rate = 0.01 + 0.9 / (1 + episode)
        random_rate = 0.05 + 0.9 / (1 + episode)
        while not maze.is_goal() and step < STEP_MAX:
            q_learn.step(LEARNING_RATE, DISCOUNT_RATE, random_rate)
            maze.draw()
            step += 1
            time.sleep(SLEEP_TIME)
            # q_learn.save_3d_log("3d_log", episode, step) 
        print(f"episode : {episode} step : {step} ")
        print("\x1b[K")
        q_learn.save_q_value(f"visualized_maze_data/q_value_episode_{episode}.txt", episode, step)   # 追加
        q_learn.plot_learning_hisotry("step_transition", episode, step) 
        # q_learn.save_3d_log_final(f"3d_log_final_{episode}")
    q_learn.save_3d_log_final_show()
        
    
if __name__ == "__main__":
    main()

# import time

# from maze import Maze
# from q_learning import QLearning


# def main():

#     EPISODE_MAX = 100
#     STEP_MAX = 3000
#     LEARNING_RATE = 0.2
#     DISCOUNT_RATE = 0.95
#     SLEEP_TIME = 0.015

#     maze = Maze()
#     q_learn = QLearning(maze)

#     for episode in range(EPISODE_MAX):
#         step = 0
#         q_learn.from_start()
#         randome_rate = 0.01 + 0.9 / (1 + episode)
#         while not maze.is_goal() and step < STEP_MAX:
#             q_learn.step(LEARNING_RATE, DISCOUNT_RATE, randome_rate)
#             maze.draw()
#             step += 1
#             time.sleep(SLEEP_TIME)
#         print(f"episode : {episode} step : {step} ")
#         print(..., end="", flush=True)  # 行末までをクリア
#         q_learn.save_q_value(f"visualized_maze_data/q_value_episode_{episode}.txt", episode, step)
#         q_learn.plot_learning_hisotry("step_transition_improved.png", episode, step)

# if __name__ == "__main__":
#     main()