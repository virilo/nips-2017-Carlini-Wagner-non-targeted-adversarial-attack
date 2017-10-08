# My Carlini-Wagner attack for NIPS-2017 non-targeted attack competition (incomplete)

It was the approach I'd have liked to do for the NIPS-2017 competition.
But it wasn't possible during the competition: https://stackoverflow.com/questions/46502291/error-getting-inceptionv3-logits-in-tensorflow
And finally, I ended up doing this solution https://github.com/virilo/nips-2017 
the last day of the competition :-/

The idea was to decrease TAU to ensure a infinite norm lower or equal to the epsilon of 16 of 256 defined by the competition. 
Once the infinite norm requisite was met, the next step would have been introducing the greater distance to the attacked image class mantaining the TAU value.




li_attack.py

	Changes the original file introducing a SAFETY_DISTANCE_K to ensure that the adversarial image has a better score than the original by a margin.


sample_li_non_targeted_MNIST.py

	Attack according to infinite norm to MNIST
	It uses a SAFETY_DISTANCE_K of 0.2

sample_li_non_targeted_Inception_V2.py

	Just to test how to configure the attack for Inception_v2

sample_li_non_targeted_Inception_V3.py

	Incomplete.  I wasn't able to configure Inception_V3: 



This code is a fork of the implementation of the attack in the paper "Towards Evaluating the Robustness of Neural Networks" by Nicholas Carlini and David Wagner.  Original code is the authors' implementation, provided under the BSD 2-Clause, Copyright 2016 to Nicholas Carlini.

https://github.com/carlini/nn_robust_attacks

Thanks the authors for the paper, result of their research, and the code.
