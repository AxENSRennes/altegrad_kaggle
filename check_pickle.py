import pickle
with open("data/validation_graphs_cached.pkl","rb") as f:
    graphs = pickle.load(f)
g0 = graphs[0]
print("spatial_pos exists:", hasattr(g0, "spatial_pos"), type(getattr(g0, "spatial_pos", None)))
print("_spatial_pos exists:", hasattr(g0, "_spatial_pos"), type(getattr(g0, "_spatial_pos", None)))