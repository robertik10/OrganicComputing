import time
import math
import mesa
import random
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


# returns all ants in current schedule as a list
def get_ants(model):
    list = model.schedule.agents
    ant_list = [agent for agent in list if isinstance(agent, AntAgent)]
    return ant_list


# returns all particles in current schedule as a list
def get_particles(model):
    list = model.schedule.agents
    particle_list = [agent for agent in list if isinstance(agent, ParticleAgent)]
    return particle_list


# rechnet die durchscnittliche Partikel Nachbarszahl von allen vorhandenen Partikeln aus
def average_particle_neighbors(model):
    # generating list with only particle neighbors
    particle_neighbor_count = [agent.neighbors for agent in model.schedule.agents if isinstance(agent, ParticleAgent)]

    average = sum(particle_neighbor_count) / len(particle_neighbor_count)

    return average


# überprüft ob alle Partikel mindestens x Nachbarn mit Distanz 0 haben
def all_have_x_neighbors(model, x):
    # generating list with only particle neighbors
    for agent in model.schedule.agents:
        if isinstance(agent, ParticleAgent) and agent.aufgehoben is False:
            count = 0
            for neighbor in model.grid.get_neighbors(agent.pos, moore=True, include_center=False):
                if isinstance(neighbor, ParticleAgent) and getDistance(agent, neighbor) == 0:
                    count += 1
            if count < x:
                return False
    return True


# zeigt ein Schaubild mit allen platzierten Partikeln
# plt.show() nachher schreiben zum anzeigen!!
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


# distanz gibt ähnlichkeit zwischen partikeln wider : Partikel sind gleich -> 0 sonst 1
def getDistance(particle1, particle2):
    distance = 0 if (particle1.type == particle2.type) else 1
    return distance


# neighborhood funktion aus dem skript
def neighborhood(i):
    alpha = 0.1
    sigma = 3
    L = []
    values = []
    neighbors = i.model.grid.get_neighborhood(
        i.pos, moore=True, include_center=False)
    for neighbor in neighbors:
        cell_content = i.model.grid.get_cell_list_contents([neighbor])
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
    result = (1 / sigma ** 2) * np.sum(values)
    return result


# pickup chance aus dem skript
def pickupChance(i):
    result = (0.1 / (0.1 + neighborhood(i))) ** 2
    return result


# drop chance aus dem skript
def dropChance(i):
    f = neighborhood(i)
    result = (f / (0.3 + f)) ** 2
    return result


# calculate entropies of given attribute
def entropy_particle_x(particles):
    length = len(particles)
    count = np.zeros(particles[0].model.grid.width)  # für jede koordinate ein Feld

    entropy = 0

    for particle in particles:
        # TODO schauen ob Spielfeld bei von 0 bis 49 geht oder 1 bis 50
        x = particle.pos[0]
        count[x] += 1



    # Anzahl der Partikel pro Koordinate in Wahrscheinlichkeit und Entropie umwandeln
    for i in range(len(count)):
        if count[i] != 0:
            p = count[i] / length
            entropy += p * np.log2(p)

    # -1 dran hängen
    return entropy * -1


def entropy_particle_y(particles):
    length = len(particles)
    count = np.zeros(particles[0].model.grid.height)  # für jede koordinate ein Feld
    entropy = 0

    for particle in particles:
        # TODO schauen ob Spielfeld bei von 0 bis 49 geht oder 1 bis 50
        x = particle.pos[1]
        count[x] += 1

    # Anzahl der Partikel pro Koordinate in Wahrscheinlichkeit und Entropie umwandeln
    for i in range(len(count)):
        if count[i] != 0:
            p = count[i] / length
            entropy += p * np.log2(p)

    # -1 dran hängen
    return entropy * -1
    return


# rechnet die entropy der Particle Neighbors aus
def entropy_particle_neighbors(particles):
    length = len(particles)
    count = np.zeros(10)  # für alle Möglichen Kombinationen an Nachbarn (0, ... , 9)
    entropy = 0

    for particle in particles:
        x = particle.neighbors
        count[x] += 1
    for i in range(len(count)):
        if count[i] != 0:
            p = count[i] / length
            entropy += p * np.log2(p)

    return entropy * -1


def entropy_ant_x(ants):
    length = len(ants)
    count = np.zeros(ants[0].model.grid.width)  # für jede koordinate ein Feld

    entropy = 0

    for agent in ants:
        # TODO schauen ob Spielfeld bei von 0 bis 49 geht oder 1 bis 50
        x = agent.pos[0]
        count[x] += 1

    # Anzahl der Partikel pro Koordinate in Wahrscheinlichkeit und Entropie umwandeln
    for i in range(len(count)):
        if count[i] != 0:
            p = count[i] / length
            entropy += p * np.log2(p)

    # -1 dran hängen
    return entropy * -1


def entropy_ant_y(ants):
    length = len(ants)
    count = np.zeros(ants[0].model.grid.height)  # für jede koordinate ein Feld

    entropy = 0

    for agent in ants:
        # TODO schauen ob Spielfeld bei von 0 bis 49 geht oder 1 bis 50
        x = agent.pos[1]
        count[x] += 1

    # Anzahl der Partikel pro Koordinate in Wahrscheinlichkeit und Entropie umwandeln
    for i in range(len(count)):
        if count[i] != 0:
            p = count[i] / length
            entropy += p * np.log2(p)

    # -1 dran hängen
    return entropy * -1


def entropy_ant_hold(ants):
    length = len(ants)
    count = np.zeros(2)  # für jede Möglichkeit ein feld also (halten oder nicht halten)

    entropy = 0

    # falls Ameisen etwas trägt -> Feld 1 oder Feld True wird um eins erhöht, sonst das Feld 0 bzw. Feld False
    for agent in ants:
        if agent.geladen is False:
            count[0] += 1
        else:
            count[1] += 1

    # Anzahl der Partikel pro Koordinate in Wahrscheinlichkeit und Entropie umwandeln
    for i in range(len(count)):
        if count[i] != 0:
            p = count[i] / length
            entropy += p * np.log2(p)

    # -1 dran hängen
    return entropy * -1


# im Folgenden werden die Emergence Methoden implementiert:
def emergence_particle_x(model, particles):
    return model.start_entropy_particle_x - entropy_particle_x(particles)


def emergence_particle_y(model, particles):
    return model.start_entropy_particle_y - entropy_particle_y(particles)


def emergence_particle_neighbors(model, particles):
    print("Start entropy neighbors: " + str(model.start_entropy_particle_neighbors))
    print("End entropy neighbors: " + str(entropy_particle_neighbors(particles)))
    return model.start_entropy_particle_neighbors - entropy_particle_neighbors(particles)


def emergence_ant_x(model, ants):
    print("Start entropy: " + str(model.start_entropy_ant_x))
    print("End entropy x: " + str(entropy_ant_x(ants)))
    return model.start_entropy_ant_x - entropy_ant_x(ants)


def emergence_ant_y(model, ants):
    return model.start_entropy_ant_y - entropy_ant_y(ants)


def emergence_ant_hold(model, ants):
    return model.start_entropy_ant_hold - entropy_ant_hold(ants)


class AntAgent(mesa.Agent):

    def __init__(self, unique_id, s, j, particle, model):
        super().__init__(unique_id, model)
        self.geladen = True
        self.particle = particle
        self.s = s
        self.j = j

        # Systemattribute

        print("Agent erstellt mit " + str(self.particle.type) + str(self.particle.pos))

    # geht einen schritt in die gewählte direction mit schrittweite s
    def schritt(self, direction):
        # direction * schrittweite s
        new_position = (self.pos[0] + direction[0] * self.s, self.pos[1] + direction[1] * self.s)

        self.model.grid.move_agent(self, new_position)
        # Particle wird mitgenommen falls vorhanden
        if self.particle is not None:
            self.model.grid.move_agent(self.particle, new_position)

    def jump(self, direction):
        for _ in range(self.j):
            self.schritt(direction)

    # step methode aus dem Skript ausimplementiert
    def step(self):
        direction = random_direction()
        self.jump(direction)

        # wenn partikel in der Hand und man auf einem Partikel landet dann wird nach stelle zum ablegen gesucht

        if self.particle is not None:
            drop = dropChance(self.particle)
            if random.random() < drop:
                if len(self.model.grid.get_cell_list_contents(self.pos)) <= 2:
                    self.particle.aufgehoben = False
                    self.particle = None
                    self.geladen = False
                    return
                # nach stelle für objekt suchen
                possible_places = self.model.grid.get_neighborhood(
                    self.pos, moore=True, include_center=False
                )
                # schauen welcher Spot kein Partikel enthält und beim ersten gefundenen Spot ablegen
                for place in possible_places:
                    # if not any(isinstance(agent, ParticleAgent) for agent in self.model.grid.get_cell_list_contents(place)):
                    if self.model.grid.is_cell_empty(place):
                        # particle an place ablegen und attribute anpassen
                        self.model.grid.move_agent(self.particle, place)  # Partikel wird hier bei place abgelegt
                        self.particle.aufgehoben = False
                        self.particle = None
                        self.geladen = False

                        break
        # alle Partikel durchgehen und mit Wahrscheinlichkeit pick aufheben
        else:
            # wenn finishing_up True ist sollen alle Ameisen ihre Partikel nur noch ablegen können, weil clustering condition erfüllt ist
            if self.model.finishing_up is True:
                return

            allParticles = [agent for agent in self.model.schedule.agents if
                            (isinstance(agent, ParticleAgent) and agent.aufgehoben is False)]
            for particle in allParticles:
                self.model.grid.move_agent(self, particle.pos)
                pick = pickupChance(particle)
                if random.random() < pick:
                    self.particle = particle
                    particle.aufgehoben = True
                    self.geladen = True
                    break


class ParticleAgent(mesa.Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, model, type):
        super().__init__(unique_id, model)
        self.aufgehoben = False
        # Systemattribute
        self.neighbors = 0
        self.x = None
        self.y = None

        if type is not None:
            self.type = type
        else:
            self.type = random.choice(["Blatt", "Stein", "Nuss"])
        if self.type == "Blatt":
            global blatt_count
            blatt_count += 1
        elif self.type == "Stein":
            global stein_count
            stein_count += 1
        else:
            global nuss_count
            nuss_count += 1

    # nach jedem step wird geschaut, wie viele benachbarte partikel man selbst hat, um das Clusteringverhalten zu beobacheten
    def step(self):
        # nach benachbarten Partikel vom selben Typ suchen,um Custering Verhalten zu beobachten
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        count = 0
        for agent in neighbors:
            if isinstance(agent, ParticleAgent) and getDistance(agent, self) == 0:
                count += 1
        self.neighbors = count

        # set systemattributes for x and y
        self.x = self.pos[0]
        self.y = self.pos[1]

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
        self.average_particle_neighbors = 0
        self.finishing_up = False  # falls die cluster condition erfüllt ist, müssen alle Ameisen ihre Partikel ablegen und dürfen keine weiteren mehr aufheben

        # systemattribute
        self.start_entropy_particle_x = 0
        self.start_entropy_particle_y = 0
        self.start_entropy_particle_neighbors = 0
        self.start_entropy_ant_x = 0
        self.start_entropy_ant_y = 0
        self.start_entropy_ant_hold = 0

        global stein_count
        stein_count = 0
        global blatt_count
        blatt_count = 0
        global nuss_count
        nuss_count = 0

        # Add the particles to grid
        k = self.num_ants + 1  # unique_identifier counter for particles
        for i in range(height):
            for j in range(width):
                # test if particle should be placed based on density parameter
                if random.random() < density:
                    a = ParticleAgent(k, self, None)  # k to ensure unique id for all agents, no specific type
                    self.schedule.add(a)
                    # Add the Ant to a grid cell
                    self.grid.place_agent(a, (i, j))
                    k += 1  # increase unique identifier

        """self.datacollector = mesa.DataCollector(
            model_reporters={"particle_neighbors": average_particle_neighbors}
        )"""

        allParticles = self.schedule.agents  # weil bisher nur Partikel zur schedule hinzugefügt wurden
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
            # Partikel und Ameise an gleichen Platz bringen und Attribute von Partikel auf angehoben anpassen
            self.grid.place_agent(a, (x, y))
            self.grid.place_agent(particle, (x, y))
            particle.aufgehoben = True

        # Wenn mehr Ameisen als Partikel eines Typs existieren, kann es passieren, dass alle gleichzeitig aufgehoben und nicht mehr abgesetzt werden können
        print(str(self.num_ants) + "Ameisen")
        while nuss_count <= self.num_ants + 1:
            print(str(nuss_count) + "Zu wenig Nüsse")
            a = ParticleAgent(k, self, "Nuss")
            self.schedule.add(a)
            self.grid.place_agent(a, self.grid.find_empty())
            k += 1
        while blatt_count <= self.num_ants + 1:
            print(str(blatt_count) + "Zu wenig Blätter")
            a = ParticleAgent(k, self, "Blatt")
            self.schedule.add(a)
            self.grid.place_agent(a, self.grid.find_empty())
            k += 1
        while stein_count <= self.num_ants + 1:
            print(str(stein_count) + "Zu wenig Steine")
            a = ParticleAgent(k, self, "Stein")
            self.schedule.add(a)
            self.grid.place_agent(a, self.grid.find_empty())
            k += 1

        allParticles = get_particles(self)
        allAnts = get_ants(self)
        # system entropies
        self.start_entropy_particle_x = entropy_particle_x(allParticles)
        print("Start entropy particle x: " + str(self.start_entropy_particle_x))
        self.start_entropy_particle_y = entropy_particle_y(allParticles)
        print("Start entropy particle y: " + str(self.start_entropy_particle_y))
        self.start_entropy_particle_neighbors = entropy_particle_neighbors(allParticles)
        print("Start entropy particle neighbors: " + str(self.start_entropy_particle_neighbors))
        self.start_entropy_ant_x = entropy_ant_x(allAnts)
        print("Start entropy ant x: " + str(self.start_entropy_ant_x))
        self.start_entropy_ant_y = entropy_ant_y(allAnts)
        print("Start entropy ant y: " + str(self.start_entropy_ant_y))
        self.start_entropy_ant_hold = entropy_ant_hold(allAnts)
        print("Start entropy ant hold: " + str(self.start_entropy_ant_hold))

        # creating all datacollectors
        self.emergence_particle_x_datacollector = mesa.DataCollector(
            model_reporters={"Emergence Particle X": [emergence_particle_x, [self, get_particles(self)]], "Emergence Particle Y": [emergence_particle_y, [self, get_particles(self)]]})
        """self.emergence_particle_y_datacollector = mesa.DataCollector(
            model_reporters={"Emergence Particle Y": [emergence_particle_y, [self, get_particles(self)]]})
        self.emergence_particle_hold_datacollector = mesa.DataCollector(
            model_reporters={"Emergence Particle Neighbors": [emergence_particle_neighbors, [self, get_particles(self)]]})
        self.emergence_ant_x_datacollector = mesa.DataCollector(
            model_reporters={"Emergence Ant X": [emergence_ant_x, [self, get_ants(self)]]})
        self.emergence_ant_y_datacollector = mesa.DataCollector(
            model_reporters={"Emergence Ant Y": [emergence_ant_y, [self, get_ants(self)]]})"""
        """self.emergence_ant_hold_datacollector = mesa.DataCollector(
            model_reporters={"Emergence Ant Hold": [emergence_ant_hold, [self, get_ants(self)]]})"""

    def step(self):
        # self.datacollector.collect(self)
        self.emergence_particle_x_datacollector.collect(self)
        """self.emergence_particle_y_datacollector.collect(self)
        self.emergence_particle_hold_datacollector.collect(self)
        self.emergence_ant_x_datacollector.collect(self)
        self.emergence_ant_y_datacollector.collect(self)"""
        #self.emergence_ant_hold_datacollector.collect(self)

        if self.finishing_up is False:
            self.schedule.step()
            if all_have_x_neighbors(self, self.cluster_cond):
                self.finishing_up = True
        else:
            # finshing_up = true -> clustering condition ist erfüllt worden
            # alle ameisen sollen jetzt ihr partikel ablegen falls noch nicht geschehen
            ants = get_ants(self)
            if any(agent.geladen for agent in ants):
                self.schedule.step()
            else:
                self.average_particle_neighbors = average_particle_neighbors(self)
                self.running = False


def test_main():
    # start timer
    start_time = time.time()

    height = 50
    width = 50
    N = 50
    cluster_cond = 3

    model = AntModel(N, 0.1, 1, 3, height, width, True, cluster_cond)
    for i in range(10000):
        if all_have_x_neighbors(model, cluster_cond):
            print("clusters have been made, after " + str(i) + " steps!")
            break
        model.step()
    particle_neighbors = model.emergence_ant_hold_datacollector.get_model_vars_dataframe()
    particle_neighbors.plot()

    plt.figure()

    show_particle_grid(model)

    # end timer
    end_time = time.time()
    print("finished in " + str(end_time - start_time) + "s!")

    plt.show()


if __name__ == "__main__":

    height = 20
    width = 20
    N = 10
    cluster_cond = 2

    model = AntModel(N, 0.1, 1, 3, height, width, False, cluster_cond)

    particles = [particle for particle in model.schedule.agents if isinstance(particle, ParticleAgent)]
    #print(particles)

    for i in range(10000):
        #print(entropy_particle_x(particles))
        if all_have_x_neighbors(model, cluster_cond):
            print("clusters have been made, after " + str(i) + " steps!")
            break
        model.step()

    particle_x = model.emergence_particle_x_datacollector.get_model_vars_dataframe()
    particle_x.plot()
    plt.figure()
