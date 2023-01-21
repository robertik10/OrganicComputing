from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import matplotlib.pyplot as plt
import numpy as np

epsilon = c.EPSILON
gamma = c.GAMMA
alpha = c.ALPHA
q_table = {}  # q learning dictionary with tuple as key and np array as values


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

        #self.mainloop()

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
    if check_is_in_q(game_grid.matrix) is False:
        new_q_entry(game_grid.matrix)


# CONTROLLER
def move(game_grid, key):

    game_grid.matrix, done = game_grid.commands[key](game_grid.matrix)

    if done:

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

# returns matrix as tuple eg: [[0,0], [2, 2]] -> ((0,0)(2,2))
def matrix_to_state(matrix):
    return tuple(tuple(row) for row in matrix)
def state_to_matrix(state):
    return [list(row) for row in state]

# returns key to press and q_value for a given state, WITHOUT checking for symmetries
def q_key_value(matrix):
    state = matrix_to_state(matrix)
    # find state q_values
    state_values = q_table.get(state)
    
    # find index of max value
    action_index = np.argmax(state_values)

    # if max 0 -> return random guess
    # otherwise return Key with the highest value
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
    

# finds state inside q_table
# returns the chosen action and the q-value of that action
# returns None if no state is found
# todo funktioniert grade nur mit 2x2 leider (nicht mal das LOLOLOLOOOOLOOL)
def lookup_q(matrix):

    if check_is_in_q(matrix) is False:
        # try to find symmetries if no state is found
        '''
        if horizontally or vertically flipped is found
         gefundener state speichern
         action, value = lookupq(gefundeneder state)
         return (flip_action(key), value)
        if rotated is found
            save amount of rotation and direction of rotation
            action, value = lookupq(gefundeneder state)
            return (rotate_action(rotations, key), value)
        '''
        return check_for_existing_symmetry(matrix)

    # if state already can be found in q_table -> return key and values based on decision process given by the task in blatt 05
    key, value = q_key_value(matrix)
    return (key, value, matrix, key)
    
def check_is_in_q(matrix):
    state = matrix_to_state(matrix)
    # find state
    state_values = q_table.get(state)
    if state_values is None:
        return False
    return True

# checks matrix for symmetry and returns needed key and value as well as the original matrix and the original key
# todo only use this as a check function
def check_for_existing_symmetry(matrix):
    h_flipped_matrix = flip_matrix_horizontally(matrix)
    check = check_is_in_q(h_flipped_matrix)
    if check is True:
        key, value = q_key_value(h_flipped_matrix)
        return (flip_action(key), value, h_flipped_matrix, key)

    y_flipped_matrix = flip_matrix_vertically(matrix)
    check = check_is_in_q(y_flipped_matrix)
    if check is True:
        key, value = q_key_value(y_flipped_matrix)
        return (flip_action(key), value, y_flipped_matrix, key)

    rotated_matrix = matrix
    for i in range(3):
        rotated_matrix = rotate_matrix_clockwise(rotated_matrix)
        check = check_is_in_q(rotated_matrix)
        if check is True:
            key, value = q_key_value(rotated_matrix)
            return (rotate_action(i,key), value, rotated_matrix, key)
    return None

def flip_matrix_horizontally(matrix):
    flipped_matrix = [[matrix[1][0], matrix[1][1]],[matrix[0][0], matrix[0][1]]]
    return flipped_matrix
def flip_matrix_vertically(matrix):
    flipped_matrix = [[matrix[0][1], matrix[0][0]],[matrix[1][1], matrix[1][0]]]
    return flipped_matrix
def rotate_matrix_clockwise(matrix):
    flipped_matrix = [[matrix[1][0], matrix[0][0]],[matrix[1][1], matrix[0][1]]]
    return flipped_matrix

# rotates action by amount of rotations needed
# rotations can be negative too: range = [-3,...,0,...,3]
# returns rotated action
# todo test if thought process is correct lol
def rotate_action(rotations, action):
    action_index = key_to_action_index(action)
    rotated_index = (action_index + rotations) % 4
    rotated_actionkey = action_index_to_key(rotated_index)
    return rotated_actionkey
# flips action to counterpart (eg: flips left to right)
def flip_action(action):
    if action == c.KEY_UP:
        return c.KEY_DOWN
    elif action == c.KEY_DOWN:
        return c.KEY_UP
    elif action == c.KEY_RIGHT:
        return c.KEY_LEFT
    elif action == c.KEY_LEFT:
        return c.KEY_RIGHT
    else : return "no valid action"


def action_index_to_key(action_index):
    if action_index == 0:
        return c.KEY_UP
    elif action_index == 1:
        return c.KEY_DOWN
    elif action_index == 2:
        return c.KEY_LEFT
    elif action_index == 3:
        return c.KEY_RIGHT
def key_to_action_index(key):
    if key == c.KEY_UP:
        return 0
    elif key == c.KEY_DOWN:
        return 1
    elif key == c.KEY_LEFT:
        return 2
    elif key == c.KEY_RIGHT:
        return 3

# inserts new state into the q-table with 0 values for all 4 actions
# if state already exists -> return None
def new_q_entry(matrix):
    if lookup_q(matrix) is not None:
        return None
    state = matrix_to_state(matrix)
    q_table[state] = np.zeros(4)

#todo jetziger state ist ein symmetrischer state der nicht gespeichert werden muss -> originaler symmertrischer state muss in q table aktualisiert werden
# schwierigkeit = welcher_action index muss aktualisiert werden? wie findet man den heraus?
# updates q_table
def update_q(matrix, new_matrix, action, reward):

    state = matrix_to_state(matrix)
    # new_state = matrix_to_state(new_matrix)

    # todo eigentlich muss nur dann ein neuer q eintrag gebildet werden, wenn symmertrischer state nicht existiert
    if check_is_in_q(new_matrix) is False:
        new_q_entry(new_matrix)

    # get index of action taken

    action_index = key_to_action_index(action)

    # update q with given formula
    # todo muss für new matrix auch schauen ob gespiegelt oder verdreht
    q_table[state][action_index] = q_table[state][action_index] + alpha * (reward + gamma * lookup_q(new_matrix)[1] - q_table[state][action_index])
    return

# new q_move
def new_q_move(game_grid):
    key = None
    # save old score and old matrix for update of q table
    old_score = logic.score
    old_matrix = game_grid.matrix


    if random.random() < epsilon:  # random choice instead of q table
        key = random.choice([c.KEY_UP,
                                   c.KEY_DOWN,
                                   c.KEY_LEFT,
                                   c.KEY_RIGHT])
    else:
        key, value, original_matrix, original_key = lookup_q(game_grid.matrix)

    move(game_grid, key)

    new_score = logic.score
    reward = new_score - old_score
    new_matrix = game_grid.matrix

    #update_q(old_matrix, new_matrix, key, new_score - old_score)
    state = matrix_to_state(old_matrix)
    # get index of action taken
    action_index = key_to_action_index(key)

    if check_is_in_q(old_matrix) is False:
        state = matrix_to_state(original_matrix)
        # get index of action taken
        action_index = key_to_action_index(original_key)

    # todo eigentlich muss nur dann ein neuer q eintrag gebildet werden, wenn symmertrischer state nicht existiert
    if check_is_in_q(new_matrix) is False:
        new_q_entry(new_matrix)



    # update q with given formula
    # todo muss für new matrix auch schauen ob gespiegelt oder verdreht
    q_table[state][action_index] = q_table[state][action_index] + alpha * (
                reward + gamma * lookup_q(new_matrix)[1] - q_table[state][action_index])
    return


# new move method used for q agent
def q_move(game_grid, key):
    # save old score and old matrix for update of q table
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
        move(game_grid, random.choice([c.KEY_UP,
                                   c.KEY_DOWN,
                                   c.KEY_LEFT,
                                   c.KEY_RIGHT]))
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


def q_simulate_1_3(sim_length, size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    game_grid = GameGrid()  # creates new game without starting it

    global epsilon
    global alpha

    # initialisation of lists for plots
    y_score = np.zeros((sim_length,))  # pre allocating list length

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

        # update score
        y_score[game_nr] = logic.score

        # update greek values
        epsilon = epsilon * c.EPSILON_DECAY
        alpha = alpha * c.ALPHA_DECAY

        # print values every 100 episodes
        if game_nr % 100 == 0 and game_nr != 0:
            print("episoden nummer: " + str(game_nr))
            print("epsilon: " + str(epsilon) + ", alpha: " + str(alpha))
            print("q size: " + str(len(q_table)))

            # calculate average score
            average = 0
            for val in y_score[game_nr - 100: game_nr - 1]: average += val
            average = average / 100
            print("average score = " + str(average))

        game_nr = game_nr + 1

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return


def q_simulate_1_4(sim_length, size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    game_grid = GameGrid()  # creates new game without starting it
    global epsilon
    global alpha

    # initialisation of lists for plots
    x = list((range(100, sim_length)))
    y_score = np.zeros((sim_length,))
    y_average = np.zeros((sim_length - 100,))

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
                key, value = lookup_q(game_grid.matrix)
                q_move(game_grid, key)

            # save new score in y_score
            y_score[game_nr] = logic.score


        # update values
        epsilon = epsilon * c.EPSILON_DECAY
        alpha = alpha * c.ALPHA_DECAY

        if game_nr % 100 == 0 and game_nr != 0:
            print("episoden nummer: " + str(game_nr))
            print("epsilon : " + str(epsilon) + ", alpha: " + str(alpha))
            print("q size: " + str(len(q_table)))

            # calculate average score
            average = 0
            for val in y_score[game_nr - 100: game_nr - 1]: average += val
            average = average / 100
            print("average score = " + str(average))

        game_nr = game_nr + 1

    # calculate averages as stated in the task description
    i = 100
    while i < sim_length:
        # average score
        average = 0
        for val in y_score[i - 100: i - 1]: average += val
        average = average / 100
        # save average in y_average
        y_average[i - 100] = average

        i = i + 1

    # plot
    #plt.subplot(2, 1, 1) wird nicht mehr verwendet
    plt.plot(x, y_average, linewidth=2.0, label='average q-agent score', color = 'green')

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return



def simulate_1_4(sim_length, size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    game_grid = GameGrid()  # creates new game without starting it

    # initialisation of lists for plots
    x = list((range(100, sim_length)))
    y_score = np.zeros((sim_length,))  # pre allocating list length
    y_average = np.zeros((sim_length - 100,))  # pre allocating list length

    # start simulation for sim_length amount of times or "games"
    game_nr = 0
    print("Running simulation with " + str(size) + "x" + str(size) + ". Might take up to a minute.")
    while game_nr < sim_length:
        reset(game_grid)
        while logic.game_state(game_grid.matrix) == 'not over':
            move(game_grid, random.choice([c.KEY_UP,
                                       c.KEY_DOWN,
                                       c.KEY_LEFT,
                                       c.KEY_RIGHT]))
        y_score[game_nr]=logic.score

        game_nr = game_nr + 1

        # print stats
        if game_nr % 100 == 0:
            print("Current Episode :" + str(game_nr) + "/" + str(sim_length))
            # calculate average score
            average = 0
            for val in y_score[game_nr - 100: game_nr - 1]: average += val
            average = average / 100
            print("average score = " + str(average))

    # calculate average scores as stated in the task description
    i = 100
    while i < sim_length:
        average = 0
        for val in y_score[i - 100: i - 1]: average += val
        average = average / 100
        # save average in y_average
        y_average[i - 100] = average
        i = i + 1

    # plot
    #plt.subplot(2, 1, 2) nicht mehr in verwendung
    plt.plot(x, y_average, linewidth=2.0, label='average random agent score', color = 'blue')
    plt.xlabel("Episodes")
    plt.ylabel("Score Average over the last 100 Episodes (stacked)")
    plt.legend()

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return

# tests if q table is being used properly
def q_test(sim_length, size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    game_grid = GameGrid()  # creates new game without starting it
    global epsilon
    global alpha

    # initialisation of lists for plots
    x = list((range(100, sim_length)))
    y_uninit_q = np.zeros((sim_length,))
    y_uninit_q_average = np.zeros((sim_length - 100,))
    y_init_q = np.zeros((sim_length,))
    y_init_q_average = np.zeros((sim_length - 100,))

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

                # update plot values for decisions based on uninit qtable values
                y_uninit_q[game_nr] += 1
            else:
                key, value = lookup_q(game_grid.matrix)
                q_move(game_grid, key)

                # update plot values for decisions based on init/uninit qtable values
                if value == 0:
                    y_uninit_q[game_nr] += 1
                else:
                    y_init_q[game_nr] += 1


        # update values
        epsilon = epsilon * c.EPSILON_DECAY
        alpha = alpha * c.ALPHA_DECAY

        if game_nr % 100 == 0 and game_nr != 0:
            print("episoden nummer: " + str(game_nr))
            print("epsilon : " + str(epsilon) + ", alpha: " + str(alpha))
            print("q size: " + str(len(q_table)))

        game_nr = game_nr + 1

    # calculate averages as stated in the task description
    i = 100
    while i < sim_length:

        # uninit q average
        uninit_average = 0
        for val in y_uninit_q[i - 100: i - 1]: uninit_average += val
        uninit_average = uninit_average / 100
        y_uninit_q_average[i - 100] = uninit_average

        # init q average
        init_average = 0
        for val in y_init_q[i - 100: i - 1]: init_average += val
        init_average = init_average / 100
        y_init_q_average[i - 100] = init_average


        i = i + 1

    plt.plot(x, y_uninit_q_average, linewidth=2.0, label='uninit q average', color='orange')
    plt.plot(x, y_init_q_average, linewidth=2.0, label='init q average', color='blue')

    plt.legend()
    plt.show()

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return

# resets qtable as well as global variables associated with it (alpha epsilon gamma)
def reset_q_learned():
    global epsilon
    global alpha
    global gamma
    q_table.clear()
    epsilon = c.EPSILON
    alpha = c.ALPHA
    gamma = c.GAMMA

# print out q table stats as stated in blatt 5 1.5
def q_stats():
    print("q-table size: "+ str(len(q_table)))
    # all states with 2x2s and 2x0s
    states = [((2,2),(0,0)),
              ((2,0),(2,0)),
              ((2,0),(0,2)),
              ((0,2),(0,2)),
              ((0,0),(2,2)),
              ((0,2),(2,0))]
    for m_state in states:
        state = matrix_to_state(m_state)
        print("State: " + str(state) + " | Best move: " + str(action_index_to_key(np.argmax(q_table.get(state)))))

def q_simulate_1_5(sim_length, size):
    c.GRID_LEN = size  # change GRID_LEN constant before starting new game
    game_grid = GameGrid()  # creates new game without starting it
    global epsilon
    global alpha

    # initialisation of lists for plots
    x = list((range(100, sim_length)))
    y_score = np.zeros((sim_length,))
    y_average = np.zeros((sim_length - 100,))

    # start simulation for sim_length amount of times or "games"
    game_nr = 0
    print("Running simulation with " + str(size) + "x" + str(size) + ". Might take a while.")
    while game_nr < sim_length:
        reset(game_grid)
        while logic.game_state(game_grid.matrix) == 'not over':
            new_q_move(game_grid)

            # save new score in y_score
            y_score[game_nr] = logic.score


        # update values
        epsilon = epsilon * c.EPSILON_DECAY
        alpha = alpha * c.ALPHA_DECAY

        if game_nr % 100 == 0 and game_nr != 0:
            print("episoden nummer: " + str(game_nr))
            print("epsilon : " + str(epsilon) + ", alpha: " + str(alpha))
            print("q size: " + str(len(q_table)))

            # calculate average score
            average = 0
            for val in y_score[game_nr - 100: game_nr - 1]: average += val
            average = average / 100
            print("average score = " + str(average))

        game_nr = game_nr + 1

    # calculate averages as stated in the task description
    i = 100
    while i < sim_length:
        # average score
        average = 0
        for val in y_score[i - 100: i - 1]: average += val
        average = average / 100
        # save average in y_average
        y_average[i - 100] = average

        i = i + 1

    # plot
    #plt.subplot(2, 1, 1) wird nicht mehr verwendet
    plt.plot(x, y_average, linewidth=2.0, label='average q-agent score', color = 'green')

    # close game after simulation ends
    try:
        game_grid.destroy()
    finally:
        return


def main_1_3(sim_length):
    q_simulate_1_3(sim_length, 2)
def main_1_4(sim_length):
    q_simulate_1_4(sim_length, 2)
    simulate_1_4(sim_length, 2)
    plt.axhline(68, color="red", label='maximum possible score')
    plt.legend()
    plt.show()
def main_1_4_q_test(sim_length):
    q_test(sim_length, 2)
def main_1_5(sim_length):
    q_simulate_1_5(sim_length, 2)
    simulate_1_4(sim_length, 2)
    plt.axhline(68, color="red", label='maximum possible score')
    plt.legend()
    plt.show()

# start simulation

#main_1_3(10000)
#q_stats()
#reset_q_learned()
#main_1_4(5000)
#reset_q_learned()
#main_1_4_q_test(10000)



main_1_5(5000)



