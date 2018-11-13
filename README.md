# TSP-GNN
Graph Neural Network architecture to solve the decision variant of the Traveling Salesperson Problem (i.e. "is there a Hamiltonian tour in G with up to a given cost"?).

OBS. To run this code you must install [pyconcorde](https://github.com/jvkersch/pyconcorde) first.

![](/figures/route-examples.png)

Upon training with -2%, +2% from the optimal cost, the model is able to achieve >80% test accuracy. It also learns to generalize for different graph distributions & larger instance sizes (with decreasing accuracy) and more relaxed deviations (with better accuracy).

The results from this experiment are reported in the research paper ["Learning to Solve NP-Complete Problems -- A Graph Neural Network for the Decision TSP"](https://arxiv.org/abs/1809.02721) by [M. Prates](http://dblp.org/pers/hd/p/Prates:Marcelo_O=_R=), [P. Avelar](http://dblp.org/pers/hd/a/Avelar:Pedro_H=_C=), [H. Lemos](http://dblp.org/pers/hd/l/Lemos:Henrique), [L. Lamb](http://dblp.org/pers/hd/l/Lamb:Lu=iacute=s_C=) and [M. Vardi](http://dblp.org/pers/hd/v/Vardi:Moshe_Y=), which has been accepted for presentation at AAAI 2019. It is available as a [arXiv.org](https://arxiv.org).

![](/figures/training-decision.png)
![](/figures/test_varying_sizes.png)
