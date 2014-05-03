NeuralNets
==========

scikit-learn friendly Implementation of Neural Nets. Will also try to reproduce results in this area

## MNIST results
Reproducing MNIST results in:

*Hinton, Geoffrey E., et al. "Improving neural networks by preventing co-adaptation of feature detectors." arXiv preprint arXiv:1207.0580 (2012).*

reproduced the results with the following two networks:

1. 784 X 800 X 800 X 10. 50% dropout on hidden units. No dropout on input units
  * can be reproduced with `python testMNIST_130.py`
2. 784 X 800 X 800 X 10. 50% dropout on both hidden and input units
  * can be reproduced with `python testMNIST_110.py`

![alt text](https://raw.githubusercontent.com/keithzhou/NeuralNets/master/result_130_vs_110.png 'Result 130 vs 110')
