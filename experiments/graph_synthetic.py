import pandas as pd
import matplotlib.pyplot as plt
a = pd.read_csv("result.csv")
a.plot(
    x="Doc M",
    y=["DESSERT", "Pytorch Brute Force Individual", "Pytorch Brute Force Combined"], 
    xlabel="m = # vectors per set", 
    ylabel="Query time (s)", 
    logx=True, 
    logy=True,
    title="Query Time v. Set Size on Synthetic Glove Data",
    colormap="tab10",
    linewidth=2.0)
plt.show()