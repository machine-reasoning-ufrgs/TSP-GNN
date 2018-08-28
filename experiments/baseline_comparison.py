
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

    SA_acc = 100*np.array([0.1436950147,0.2099609375,0.28125,0.341796875,0.40234375,0.474609375,0.5498046875,0.6181640625,0.681640625,0.734375])
    NN_acc = 100*np.array([0.00390625,0.0068359375,0.01171875,0.01953125,0.029296875,0.0478515625,0.07421875,0.0966796875,0.1220703125,0.1552734375])

    # Create figure
    figure, axis = plt.subplots(figsize=(4,3))

    # Set axes' labels
    axis.set_xlabel('Deviation (%)')
    axis.set_ylabel('True Positive Rate (%)')

    # Set x-axis ticks
    axis.set_xticks(np.arange(1,10+1))

    # Draw guide lines
    for y in range(0,100+1,10):
        axis.axhline(y=y, linewidth=0.75, color='gray', zorder=2)
    #end

    with open('results/test_varying_dev_TP.dat') as f:
        data = np.array([ [float(x) for x in line.split()] for line in f.readlines()])
        axis.plot(100*data[:,0],100*data[:,1], marker='o', mfc='white', color='blue', label='GNN')
    #end

    axis.plot(np.linspace(0,10,11)[1:],SA_acc, marker='o', mfc='white', color='green', label='SA')
    axis.plot(np.linspace(0,10,11)[1:],NN_acc, marker='o', mfc='white', color='red', label='NN')

    axis.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig('figures/test_varying_dev_baseline.eps', format='eps')

#end