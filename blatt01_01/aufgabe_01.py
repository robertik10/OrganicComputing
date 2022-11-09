import mesa
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def count_particle_neighbours():
    #TODO
    '''particle_neighbours = [agent.neighbours for agent in model.schedule.agents]
    x = filter(lambda instance: isinstance(instance, ParticleAgent) ,sorted(particle_neighbours))

    print(x)
    return x'''
    return


class AntAgent(mesa.Agent):

    def __init__(self, unique_id, s, j, model):
        super().__init__(unique_id, model)
        self.geladen = False
        self.particle = None
        self.s = s
        self.j = j

    # gibt zufällige direction zurück
    def random_direction(self):
        # direction ist ein tupel das die richtung in x,y Koordinaten angibt also: (-1, 0), (1, 0), (0, -1), (0, 1)
        direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
        return direction

    # geht einen schritt in die gewählte direction mit schrottweite s
    def schritt(self, direction):
        # direction * schrittweite s
        new_position = (self.pos[0] + direction[0] * self.s, self.pos[1] + direction[1] * self.s)

        self.model.grid.move_agent(self, new_position)
        # Particle wird mitgenommen falls vorhanden
        if self.particle is True:
            self.model.grid.move_agent(self.particle, new_position)

        print(self.pos)


    def jump(self, direction):
        for i in range(self.j):
            self.schritt(direction)

    def step(self):
        # find cell content of own cell
        cell_content = self.model.grid.get_cell_list_contents([self.pos])

        particle = None
        # find particle that is not aufgehoben
        for x in cell_content:
            if isinstance(x, ParticleAgent) and x.aufgehoben is False:
                particle = x
                break

        if not self.geladen and particle is not None:
            # Attribute setzen
            self.particle = particle
            self.geladen = True
            particle.aufgehoben = True

            print("servus i bims Ameise " + str(self.unique_id)  + " und hab was aufgehoben hihi")

            # random direction finden und dann in diese richtung jumpen
            direction = self.random_direction()
            self.jump(direction)

        elif self.geladen and particle is not None:
            # nach stelle für objekt suchen
            possible_places = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=False
            )
            # schauen welcher Spot kein Partikel enthält und beim ersten gefundenen Spot ablegen
            for place in possible_places:
                # TODO rauslassen : if not any(isinstance(self.model.grid.get_cell_list_contents(place), ParticleAgent))
                if self.model.grid.is_cell_empty(place):
                    # particle an place ablegen und attribute anpassen
                    self.model.grid.move_agent(self.particle, place) # Partikel wird hier bei place abgelegt
                    self.particle.aufgehoben = False
                    self.particle = None
                    self.geladen = False


                    print("servus i bims Ameise " + str(self.unique_id) + " hab was abgelegt :(")


                    break
            # random direction finden und dann in diese richtung jumpen auch wenn kein freier platz gefunden wurde, um "gefangen sein" zu vermeiden
            direction = self.random_direction()
            self.jump(direction)

        else:
            direction = self.random_direction()
            self.schritt(direction)


class ParticleAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.aufgehoben = False
        self.neighbours = 0

    def step(self):
        # nach benachbartem Partikel suchen,um Custering Verhalten zu beobachten
        neighbours = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        # zählen wie viele partikel an uns angrenzen
        count = 0
        for place in neighbours:
            if not isinstance(self.model.grid.get_cell_list_contents(place), ParticleAgent):
                count += 1

        self.neighbours = count
        return


class AntModel(mesa.Model):
    """A model with some number of agents."""

    # middleInit is boolean value
    def __init__(self, N, density, s, j, height, width, middleInit):

        self.num_ants = N
        self.grid = mesa.space.MultiGrid(height, width, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True

        # Create agents
        for i in range(self.num_ants):
            a = AntAgent(i, s, j, self)
            self.schedule.add(a)

            # if middleInit is True place all ants in the middle, else randomized placement
            if middleInit is True:
                x = self.grid.width // 2
                y = self.grid.height // 2
            else:
                # Add the Ant to a random grid cell
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        # Add the particles to grid
        k = N + 1 #unique_identifier counter for particles
        for i in range(height):
            for j in range(width):
                # test if particle should be placed based on density parameter
                if random.random() < density:
                    a = ParticleAgent(k, self) #k to ensure unique id for all agents
                    self.schedule.add(a)
                    # Add the Ant to a grid cell
                    self.grid.place_agent(a, (i, j))
                    k += 1  #increase unique identifier
        #TODO
        """self.datacollector = mesa.DataCollector(
            model_reporters={"particle_neighbours":count_particle_neighbours}
        )"""

    def step(self):
        #TODO self.datacollector.collect(self)
        self.schedule.step()

def batch_run():
    params = {"N": 10, "density":0.1, "s":1, "j":3, "height": 10, "width": 10, "middleInit": (True, False)}

    results = mesa.batch_run(
        AntModel,
        parameters=params,
        iterations=2,
        max_steps=1000,
        number_processes=1,
        data_collection_period=1,
        display_progress=True
    )

    results_df = pd.DataFrame(results)
    print(results_df.keys())



def firt_main():
    height = 30
    width = 30
    model = AntModel(100, 0.1, 1, 5, height, width, True)

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    plt.imshow(agent_counts, interpolation="nearest")
    plt.colorbar()
    # necessary to show plot
    plt.show()

    for i in range(50000):
        model.step()

    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        agent_count = len(cell_content)
        agent_counts[x][y] = agent_count

    plt.imshow(agent_counts, interpolation="nearest")
    plt.colorbar()
    # necessary to show plot
    plt.show()

if __name__ == "__main__":
    height = 30
    width = 30
    model = AntModel(10, 0.1, 1, 5, height, width, True)

    for i in range(5000):
        model.step()
    #todo
    #particle_neighbours = model.datacollector.get_model_vars_dataframe()
    #particle_neighbours.plot()

    """end_neighbours = particle_neighbours.xs(99, level="Step")["Wealth"]
    end_neighbours.hist(bins=range(particle_neighbours.neighbours.max() + 1))"""
    plt.show()

