"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # self.print_outlier(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    def print_outlier(self, x):

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111,projection='3d')

        for i in range(x.shape[1]):
            # Generate the random data for the y=k 'layer'.
            ys = range(x.shape[-1])

            # You can provide either a single color or an array with the same length as
            # xs and ys. To demonstrate this, we color the first bar of each set cyan.
            max_value = int(x[:,i,:].max().round())
            if max_value >= 1:
                cs = ["r"] * max_value
            else:
                cs = ["b"] * max_value+1
            cs[0] = 'c'

            # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
            ax.bar(ys, x[:,i,:].flatten().cpu(),width=40, zs=i, zdir='y', color=cs, alpha=0.8)

        ax.set_xlabel("Channels")
        ax.set_ylabel("Tokens")
        ax.set_zlabel("Absolute Value")
        ax.set_title("Activation (Original)\nHard to Quantize")

        # On the y-axis let's only label the discrete values that we have data for.
        ax.set_yticks(range(x.shape[1]))
        path = 1
        while os.path.isfile(str(path)+".png"):
            path += 1
        plt.savefig(str(path)+".png", dpi=300, bbox_inches='tight')
        