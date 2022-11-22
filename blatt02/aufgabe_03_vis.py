from mesa.visualization.ModularVisualization import ModularServer

from aufgabe_03 import *
import mesa

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}

    if isinstance(agent, ParticleAgent) and agent.type == "Nuss":
        portrayal["Color"] = "brown"
        portrayal["Layer"] = 1
    elif isinstance(agent, ParticleAgent) and agent.type == "Blatt":
        portrayal["Color"] = "green"
        portrayal["Layer"] = 1
        #portrayal["r"] = 0.2
    elif isinstance(agent, ParticleAgent) and agent.type == "Stein":
        portrayal["Color"] = "grey"
        portrayal["Layer"] = 1
        #portrayal["r"] = 0.2
    else:
        portrayal["Color"] = "black"
        portrayal["Layer"] = 2
        portrayal["r"] = 0.2

    return portrayal

if __name__ == "__main__":

    width = 20
    hight = 20
    N = 5
    s = 2
    j = 5


    grid = mesa.visualization.CanvasGrid(agent_portrayal, width, hight, 100, 100)
    server = mesa.visualization.ModularServer(
        AntModel, [grid], "Ant Model",
        {"N": N, "density": 0.1, "s": s, "j": j, "width": width, "height": hight, "middleInit": False, "cluster_cond": 3}
    )

    chart_particle_x = mesa.visualization.ChartModule([{"Label": "Emergenz Particle X",
                                                        "Color":"Red"}],
                                                      data_collector_name ='emergence_particle_x_datacollector')

    server = ModularServer(AntModel,
                           [grid, chart_particle_x],
                           "And Model",
                           {"N": N, "density": 0.1, "s": s, "j": j, "width": width, "height": hight, "middleInit": False,
                            "cluster_cond": 2})
    server.port = 8521  # The default
    server.launch()

