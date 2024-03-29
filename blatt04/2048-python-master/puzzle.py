from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import matplotlib.pyplot as plt


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


# CONTROLLER
def move(game_grid):
    while logic.game_state(game_grid.matrix) == 'not over':
        rand_move = random.choice([c.KEY_UP,
                                   c.KEY_DOWN,
                                   c.KEY_LEFT,
                                   c.KEY_RIGHT])
        game_grid.matrix, done = game_grid.commands[rand_move](game_grid.matrix)
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


# O/C simulation with size determining the frame size -> e.g. size = 4 -> Frame is 4x4
def simulate(size):
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


# start simulation
simulate(2)
simulate(3)
simulate(4)
