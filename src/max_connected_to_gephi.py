import networkx as nx
import operator
import pandas as pd
import numpy as np

if __name__ == "__main__":
  G = nx.read_gpickle("/home/kapxy/networkanalysis/OverflowNA/data/Users_comments_nanremoved_noselfloops.pkl")
  print(len(G.nodes))
  centralities = nx.degree_centrality(G)
  max_centrality = max(centralities.items(), key=operator.itemgetter(1))[0]
  print(nx.degree(G, max_centrality))
  neighbors = list(nx.neighbors(G, max_centrality))
  neighbors_of_neighbors = { max_centrality: neighbors }
  
  for neighbor in neighbors:
    neighbors_of_neighbors[neighbor] = list(nx.neighbors(G, neighbor))
  #print(neighbors_of_neighbors)
  thonk = np.array([y for x in [[(k, target) for target in v] for k, v in neighbors_of_neighbors.items()] for y in x])
  print(thonk[14])
  df = pd.DataFrame(np.array(thonk), columns=["Source", "Target"])
  df.to_csv("data/two_step_neighbors_from_max_degree_centrality.csv", index=False)
  # nx.write_edgelist(max_conn, path="data/max_conn_edgelist.csv", delimiter=",", data=False)
