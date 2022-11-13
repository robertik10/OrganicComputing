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



    grid = mesa.visualization.CanvasGrid(agent_portrayal, 15, 15, 500, 500)
    server = mesa.visualization.ModularServer(
        AntModel, [grid], "Ant Model",
        {"N": 10, "density": 0.1, "s": 2, "j": 3, "width": 15, "height": 15, "middleInit": False, "cluster_cond": 2}
    )
    server = ModularServer(AntModel,
                           [grid],
                           "And Model",
                           {"N": 10, "density": 0.1, "s": 2, "j": 3, "width": 15, "height": 15, "middleInit": False,
                            "cluster_cond": 2})
    server.port = 8521  # The default
    server.launch()

