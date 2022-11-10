from mesa.visualization.ModularVisualization import ModularServer

from aufgabe_01 import *
import mesa

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}

    if isinstance(agent, ParticleAgent):
        portrayal["Color"] = "red"
        portrayal["Layer"] = 0
    else:
        portrayal["Color"] = "grey"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.2

    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, 10, 10, 500, 500)
server = mesa.visualization.ModularServer(
    AntModel, [grid], "Ant Model", {"N": 10, "density":0.3, "s":1, "j":3, "width": 10, "height": 10, "middleInit": True, "cluster_cond": 3}
)
server = ModularServer(AntModel,
                       [grid],
                       "Money Model",
                       {"N": 10, "density":0.3, "s":1, "j":3, "width":10, "height":10, "middleInit": True, "cluster_cond": 3})
server.port = 8521 # The default
server.launch()