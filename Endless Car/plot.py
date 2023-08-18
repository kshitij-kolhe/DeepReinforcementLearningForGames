import matplotlib.pyplot as plt
from IPython import display

class Plot():
    def __init__(self):
        plt.ion()
        return
    
    def plot(self, name: str, val1:[], val2:[]):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.clf()
        plt.title(name)
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(val1, label = "score")
        plt.plot(val2, label = "mean score")
        plt.ylim(ymin=0)
        plt.text(len(val1)-1, val1[-1], str(val1[-1]))
        plt.text(len(val2)-1, val2[-1], str(val2[-1]))
        leg = plt.legend(loc='upper center')
        plt.show(block=False)
        plt.pause(.1)