from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import matplotlib.pyplot as plt
import numpy as np

epsilon = 1
gamma = 1
alpha = 1
q_table = {}  # q learning dictionary


def gen():
    return random.randint(0, c.GRID_LEN - 1)


class GameGrid(Frame):
    def __init__(self):
        Frame.__init__(self)
        self.grid()
        self.master.title('2048')
        self.master.bind("<Key>", move)

        self.commands = {
            c.KEY_UP: logic.up,
            c.KEY_DOWN: logic.down,
            c.KEY_LEFT: logic.left,
            c.KEY_RIGHT: logic.right,
            c.KEY_UP_ALT1: logic.up,
            c.KEY_DOWN_ALT1: logic.down,
            c.KEY_LEFT_ALT1: logic.left,
            c.KEY_RIGHT_ALT1: logic.right,
            c.KEY_UP_ALT2: logic.up,
            c.KEY_DOWN_ALT2: logic.down,
            c.KEY_LEFT_ALT2: logic.left,
            c.KEY_RIGHT_ALT2: logic.right,
        }

        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrixs = []
        self.update_grid_cells()

        # self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()

        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(
                    background,
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    width=c.SIZE / c.GRID_LEN,
                    height=c.SIZE / c.GRID_LEN
                )
                cell.grid(
                    row=i,
                    column=j,
                    padx=c.GRID_PADDING,
                    pady=c.GRID_PADDING
                )
                t = Label(
                    master=cell,
                    text="",
                    bg=c.BACKGROUND_COLOR_CELL_EMPTY,
                    justify=CENTER,
                    font=c.FONT,
                    width=5,
                    height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(
                        text=str(new_number),
                        bg=c.BACKGROUND_COLOR_DICT[new_number],
                        fg=c.CELL_COLOR_DICT[new_number]
                    )
        self.update_idletasks()

    # old move function -> not in use right now
    def key_down(self, event):
        key = event.keysym
        print(event)
        if key == c.KEY_RESET: self.reset()
        if key == c.KEY_QUIT: exit()
        if key == c.KEY_BACK and len(self.history_matrixs) > 1:
            self.matrix = self.history_matrixs.pop()
            self.update_grid_cells()
            print('back on step total step:', len(self.history_matrixs))
        elif key in self.commands:
            self.matrix, done = self.commands[key](self.matrix)
            if done:
                self.matrix = logic.add_two(self.matrix)
                # record last move
                self.history_matrixs.append(self.matrix)
                self.update_grid_cells()
                if logic.game_state(self.matrix) == 'win':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                if logic.game_state(self.matrix) == 'lose':
                    self.grid_cells[1][1].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                    self.grid_cells[1][2].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)

    def generate_next(self):
        index = (gen(), gen())
        while self.matrix[index[0]][index[1]] != 0:
            index = (gen(), gen())
        self.matrix[index[0]][index[1]] = 2


# OBSERVER
def state(game_grid):
    # find out if game_grid is finished
    if logic.game_state(game_grid.matrix) == 'win' or logic.game_state(game_grid.matrix) == 'lose':
        finished = True
    else:
        finished = False

    # matrix of game_grid
    matrix = game_grid.matrix

    # game_grid points
    current_score = logic.score

    return matrix, current_score, finished


# CONTROLLER
def reset(game_grid):
    logic.update_score(0)
    game_grid.matrix = logic.new_game(c.GRID_LEN)
    game_grid.history_matrixs = []
    game_grid.update_grid_cells()

    # todo vielleicht in simulation reinschreiben und nicht hier
    # update q_table
    if lookup_q(game_grid.matrix) is None:
        new_q_entry(game_grid.matrix)


# CONTROLLER
# verbessert
def move(game_grid, key):
    # old_score = logic.score
    # old_matrix = game_grid.matrix

    game_grid.matrix, done = game_grid.commands[key](game_grid.matrix)

    if done:
        # new_score = logic.score
        # new_matrix = game_grid.matrix
        # update_q(old_matrix, new_matrix, key, new_score - old_score)

        game_grid.matrix = logic.add_two(game_grid.matrix)
        # record last move
        game_grid.history_matrixs.append(game_grid.matrix)
        game_grid.update_grid_cells()
        if logic.game_state(game_grid.matrix) == 'win':
            game_grid.grid_cells[1][0].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            game_grid.grid_cells[1][1].configure(text="Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
        if logic.game_state(game_grid.matrix) == 'lose':
            game_grid.grid_cells[1][0].configure(text="You", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
            game_grid.grid_cells[1][1].configure(text="Lose!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)


def matrix_to_state(matrix):
    return tuple(tuple(row) for row in matrix)


# finds state inside q_table
# returns the chosen action and the qvalue of that action
# returns None if no state is found
def lookup_q(matrix):
    state = matrix_to_state(matrix)
    # find state
    state_values = q_table.get(state)
    if state_values is None: return None  # return None if not found

    # find index of max value
    action_index = np.argmax(state_values)

    # if max 0 -> return random guess
    # otherwise return Key with highest value
    if state_values[action_index] == 0:
        return random.choice([c.KEY_UP,
                              c.KEY_DOWN,
                              c.KEY_LEFT,
                              c.KEY_RIGHT]), 0
    elif action_index == 0:
        return c.KEY_UP, state_values[0]
    elif action_index == 1:
        return c.KEY_DOWN, state_values[1]
    elif action_index == 2:
        return c.KEY_LEFT, state_values[2]
    elif action_index == 3:
        return c.KEY_RIGHT, state_values[3]


# inserts new state into the qtable with 0 values for all 4 actions
# if state already exists -> return None
def new_q_entry(matrix):
    if lookup_q(matrix) is not None:
        return None
    state = matrix_to_state(matrix)
    q_table[state] = np.zeros(4)


# update q_table
def update_q(matrix, new_matrix, action, reward):
    state = matrix_to_state(matrix)
    # new_state = matrix_to_state(new_matrix)

    if lookup_q(new_matrix) is None:
        new_q_entry(new_matrix)

    # get index of action taken
    action_index = None
    if action == c.KEY_UP:
        action_index = 0
    elif action == c.KEY_DOWN:
        action_index = 1
    elif action == c.KEY_LEFT:
        action_index = 2
    elif action == c.KEY_RIGHT:
        action_index = 3

    # print("statepos : " + str(state_index))
    # print("actpos : " + str(action_index))
    q_table[state][action_index] = q_table[state][action_index] + alpha * (
            reward + gamma * lookup_q(new_matrix)[1] - q_table[state][action_index])
    return


def q_move(game_grid, key):
    old_score = logic.score
    old_matrix = game_grid.matrix

    move(game_grid, key)

    new_score = logic.score
    new_matrix = game_grid.matrix
    update_q(old_matrix, new_matrix, key, new_score - old_score)


# O/C simulation with size determining the frame size -> e.g. size = 4 -> Frame is 4x4
def old_simulate(size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    sim_length = 100  # set simulation length
    game_grid = GameGrid()  # creates new game without starting it

    # initialisation of lists for plots
    x = list((range(0, sim_length)))
    y_score = []


    # start simulation for sim_length amount of times or "games"
    game_nr = 0
    print("Running simulation with " + str(size) + "x" + str(size) + ". Might take up to a minute.")
    while game_nr < sim_length:
        reset(game_grid)
        move(game_grid)
        y_score.append(logic.score)

        game_nr = game_nr + 1
        print("Current Game :" + str(game_nr) + "/" + str(sim_length))

    # calculate average score
    average = 0
    for val in y_score: average += val
    average = average / sim_length
    print("average score = " + str(average))

    # plot
    plt.plot(x, y_score, linewidth=2.0, label='score')
    plt.axhline(average, color="red", label='average')
    plt.legend()
    plt.show()

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return


def q_simulate_1_3(size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    sim_length = 10000  # set simulation length
    game_grid = GameGrid()  # creates new game without starting it
    global epsilon
    global alpha

    # start simulation for sim_length amount of times or "games"
    game_nr = 0
    print("Running simulation with " + str(size) + "x" + str(size) + ". Might take a while.")
    while game_nr < sim_length:
        reset(game_grid)
        while logic.game_state(game_grid.matrix) == 'not over':
            if random.random() < epsilon:  # random choice instead of q table
                rand_move = random.choice([c.KEY_UP,
                                           c.KEY_DOWN,
                                           c.KEY_LEFT,
                                           c.KEY_RIGHT])
                q_move(game_grid, rand_move)
            else:
                key = lookup_q(game_grid.matrix)[0]
                q_move(game_grid, key)

        # update greek values
        epsilon = epsilon * 0.9
        alpha = alpha * 0.9

        # print values every 100 episodes
        if game_nr % 100 == 0:
            print("episoden nummer: " + str(game_nr))
            print("epsilon : " + str(epsilon) + ", alpha: " + str(alpha))
            print("q size: " + str(len(q_table)))
            print("score: " + str(logic.score))

        game_nr = game_nr + 1

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return


def q_simulate_1_4(size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    sim_length = 10000  # set simulation length
    game_grid = GameGrid()  # creates new game without starting it
    global epsilon
    global alpha

    # initialisation of lists for plots
    x = list((range(100, sim_length + 1)))
    y_score = np.zeros((sim_length,)) #pre allocating list length
    y_average = np.zeros((sim_length - 99,)) #pre allocating list length

    # start simulation for sim_length amount of times or "games"
    game_nr = 0
    print("Running simulation with " + str(size) + "x" + str(size) + ". Might take a while.")
    while game_nr < sim_length:
        reset(game_grid)
        while logic.game_state(game_grid.matrix) == 'not over':
            if random.random() < epsilon:  # random choice instead of q table
                rand_move = random.choice([c.KEY_UP,
                                           c.KEY_DOWN,
                                           c.KEY_LEFT,
                                           c.KEY_RIGHT])
                q_move(game_grid, rand_move)
            else:
                key = lookup_q(game_grid.matrix)[0]
                q_move(game_grid, key)

            # save new score in y_score
            y_score[game_nr] = logic.score

        # update values
        epsilon = epsilon * 0.9  # todo als konstante einstellen
        alpha = alpha * 0.9  # todo als konstante einstellen
        if game_nr % 100 == 0:
            print("episoden nummer: " + str(game_nr))
            print("epsilon : " + str(epsilon) + ", alpha: " + str(alpha))
            print("q size: " + str(len(q_table)))
            print("score: " + str(logic.score))

        game_nr = game_nr + 1

    # calculate average scores as stated in the task description
    i = 100
    while i <= sim_length:
        average = 0
        for val in y_score[i - 100: i]: average += val
        average = average / 100
        # save average in y_average
        y_average[i - 100] = average
        i = i + 1

    # plot
    plt.plot(x, y_average, linewidth=2.0, label='average score')
    #plt.axhline(average, color="red", label='average')
    plt.legend()
    plt.show()

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return


def simulate_1_4(size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    sim_length = 10000  # set simulation length
    game_grid = GameGrid()  # creates new game without starting it

    # initialisation of lists for plots
    x = list((range(100, sim_length + 1)))
    y_score = np.zeros((sim_length,))  # pre allocating list length
    y_average = np.zeros((sim_length - 99,))  # pre allocating list length

    # start simulation for sim_length amount of times or "games"
    game_nr = 0
    print("Running simulation with " + str(size) + "x" + str(size) + ". Might take up to a minute.")
    while game_nr < sim_length:
        reset(game_grid)
        move(game_grid)
        y_score[game_nr]=logic.score

        game_nr = game_nr + 1
        print("Current Game :" + str(game_nr) + "/" + str(sim_length))

    # calculate average scores as stated in the task description
    i = 100
    while i <= sim_length:
        average = 0
        for val in y_score[i - 100: i]: average += val
        average = average / 100
        # save average in y_average
        y_average[i - 100] = average
        i = i + 1

    # plot
    plt.plot(x, y_average, linewidth=2.0, label='average score')
    # plt.axhline(average, color="red", label='average')
    plt.legend()
    plt.show()

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return


# start simulation
# simulate(2)
# simulate(3)
# simulate(4)
# q_simulate_1_3(2)
q_simulate_1_4(2)
simulate_1_4(2)
