from mesa.visualization.ModularVisualization import ModularServer

from aufgabe_02 import *
import mesa

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0,
                 "Color": "red",
                 "r": 0.5}

    if isinstance(agent, ParticleAgent) and agent.type == "Nuss":
        portrayal["Color"] = "brown"
        portrayal["Layer"] = 0
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
        portrayal["Layer"] = 1
        portrayal["r"] = 0.2

    return portrayal

grid = mesa.visualization.CanvasGrid(agent_portrayal, 10, 10, 500, 500)
server = mesa.visualization.ModularServer(
    AntModel, [grid], "Ant Model", {"N": 1, "density":0.3, "s":1, "j":3, "width": 10, "height": 10, "middleInit": True, "cluster_cond": 3}
)
server = ModularServer(AntModel,
                       [grid],
                       "And Model",
                       {"N": 1, "density":0.3, "s":1, "j":3, "width":10, "height":10, "middleInit": True, "cluster_cond": 3})
server.port = 8521 # The default
server.launch()