import time

import mesa
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# rechnet die durchscnittliche Partikel Nachbarszahl von allen vorhandenen Partikeln aus
def average_particle_neighbours(model):
    # generating list with only particle neighbours
    particle_neighbour_count = [agent.neighbours for agent in model.schedule.agents if isinstance(agent, ParticleAgent)]

    average = sum(particle_neighbour_count) / len(particle_neighbour_count)

    return average


# überprüft ob alle Partikel mindestens x Nachbarn haben
def all_have_x_neighbours(model, x):
    # generating list with only particle neighbours
    particle_neighbour_count = [agent.neighbours for agent in model.schedule.agents if isinstance(agent, ParticleAgent)]
    if any(count < x for count in particle_neighbour_count):
        return False
    else:
        return True


# zeigt ein Schaubild mit allen platzierten Partikeln
# !!unbedingt plt.show() nachher schreiben zum anzeigen!!
def show_particle_grid(model):
    agent_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, x, y = cell
        filtered_cell_content = [x for x in cell_content if
                                 isinstance(x, ParticleAgent)]  # filtering out all the ants in the data
        agent_count = len(filtered_cell_content)
        agent_counts[x][y] = agent_count

    plt.imshow(agent_counts, interpolation="nearest")
    plt.colorbar()


# gibt zufällige direction als tupel zurück
def random_direction():
    # direction ist ein tupel das die richtung in x,y Koordinaten angibt also: (-1, 0), (1, 0), (0, -1), (0, 1)
    direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    return direction


def getDistance(particle1, particle2):
    distance = 0 if (particle1.type == particle2.type) else 1
    return distance


def neighborhood(i):
    alpha = 0.01
    sigma = 9
    L = []
    values = []
    neighbors = model.grid.get_neighborhood(
        i.pos, moore=True, include_center=False)
    for neighbor in neighbors:
        cell_content = model.grid.get_cell_list_contents([neighbor])
        for x in cell_content:
            if isinstance(x, ParticleAgent) and x.aufgehoben is False:
                L.append(x)
                break
    for j in L:
        value = 1 - (getDistance(i, j) / alpha)
        if value > 0:
            values.append(value)
        else:
            return 0
    result = (1 / sigma) * np.sum(values)
    return result


def pickupChance(i):
    result = (0.1 / (0.1 + neighborhood(i))) ** 2
    return result


def dropChance(i):
    f = neighborhood(i)
    result = (f / (0.3 + f)) ** 2
    return result


class AntAgent(mesa.Agent):

    def __init__(self, unique_id, s, j, particle, model):
        super().__init__(unique_id, model)
        self.geladen = True
        self.particle = particle
        self.s = s
        self.j = j
        print("Agent erstellt mit " + str(self.particle.type) + str(self.particle.pos))

    # geht einen schritt in die gewählte direction mit schrittweite s
    def schritt(self, direction):
        # direction * schrittweite s
        new_position = (self.pos[0] + direction[0] * self.s, self.pos[1] + direction[1] * self.s)

        self.model.grid.move_agent(self, new_position)
        # Particle wird mitgenommen falls vorhanden
        if self.particle is True:
            self.model.grid.move_agent(self.particle, new_position)

    def jump(self, direction):
        for i in range(self.j):
            self.schritt(direction)

    def step(self):
        direction = random_direction()
        self.jump(direction)
        drop = dropChance(self.particle)
        if random.random() < drop:
            # nach stelle für objekt suchen
            possible_places = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=True
            )
            # schauen welcher Spot kein Partikel enthält und beim ersten gefundenen Spot ablegen
            for place in possible_places:
                # if not any(isinstance(agent, ParticleAgent) for agent in self.model.grid.get_cell_list_contents(place)):
                if self.model.grid.is_cell_empty(place):
                    # particle an place ablegen und attribute anpassen
                    print(str(self.particle.type) + " abgelegt auf " + str(place) + str(drop))
                    self.model.grid.move_agent(self.particle, place)  # Partikel wird hier bei place abgelegt
                    self.particle.aufgehoben = False
                    self.particle = None
                    self.geladen = False

                    break

            allParticles = [agent for agent in self.model.schedule.agents if
                            (isinstance(agent, ParticleAgent) and agent.aufgehoben is False)]
            while self.geladen is False:
                particle = random.choice(allParticles)
                pick = pickupChance(particle)
                if random.random() < pick:
                    self.particle = particle
                    particle.aufgehoben = True
                    self.geladen = True
                    print("Particle aufgehoben auf " + str(self.pos) + str(pick))


class ParticleAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.aufgehoben = False
        self.neighbours = 0
        self.type = random.choice(["Blatt", "Stein", "Nuss"])

    # nach jedem step wird geschaut, wie viele benachbarte partikel man selbst hat, um das Clusteringverhalten zu beobacheten
    def step(self):
        # nach benachbarten Partikel suchen,um Custering Verhalten zu beobachten
        neighbours = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        # schauen was auf den Nachbarsplätzen liegt
        neighbours_content = self.model.grid.get_cell_list_contents(neighbours)

        # nur Zellen mit Partikeln rausfiltern
        filtered_content = [agent for agent in neighbours_content if isinstance(agent, ParticleAgent)]

        # zählen wie viele partikel an uns angrenzen
        count = 0
        for place in filtered_content:
            count += 1

        self.neighbours = count
        return


class AntModel(mesa.Model):
    """A model with some number of agents."""

    # middleInit is boolean value
    # cluster_cond ist ein integer und die Bedingung zum abbrechen der steps. (bei cluster_cond anzahl an partikel nachbarn solls aufhören)
    def __init__(self, N, density, s, j, height, width, middleInit, cluster_cond):

        self.num_ants = N
        self.grid = mesa.space.MultiGrid(height, width, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.cluster_cond = cluster_cond
        self.average_particle_neighbours = 0

        # Add the particles to grid
        k = N + 1  # unique_identifier counter for particles
        for i in range(height):
            for j in range(width):
                # test if particle should be placed based on density parameter
                if random.random() < density:
                    a = ParticleAgent(k, self)  # k to ensure unique id for all agents
                    self.schedule.add(a)
                    # Add the Ant to a grid cell
                    self.grid.place_agent(a, (i, j))
                    k += 1  # increase unique identifier

        self.datacollector = mesa.DataCollector(
            model_reporters={"particle_neighbours": average_particle_neighbours}
        )

        allParticles = self.schedule.agents
        # Create agents
        for i in range(self.num_ants):
            # select random particle and pick it up
            particle = random.choice(allParticles)
            allParticles.remove(particle)

            a = AntAgent(i, s, j, particle, self)
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

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if all_have_x_neighbours(self, self.cluster_cond):
            print("clusters have been made, afer " + '''str(step_count)''' + " steps!")
            self.average_particle_neighbours = average_particle_neighbours(self)
            self.running = False


if __name__ == "__main__":
    # start timer
    start_time = time.time()

    height = 10
    width = 10
    cluster_cond = 2

    model = AntModel(5, 0.2, 1, 3, height, width, True, cluster_cond)
    step_count = 0
    for i in range(10000):
        if all_have_x_neighbours(model, cluster_cond):
            print("clusters have been made, after " + str(step_count) + " steps!")
            break
        model.step()
        step_count += 1

    particle_neighbours = model.datacollector.get_model_vars_dataframe()
    particle_neighbours.plot()

    plt.figure()

    show_particle_grid(model)

    # end timer
    end_time = time.time()
    print("finished in " + str(end_time - start_time) + "s!")

    # nach dem timer erst weil plt.show blockierend wirkt
    plt.show()
    # batch_run()
