# Custom-PyTorch-Resources
 Classes and functions for hidden unit activation functions, regularization, and pruning. Some are implementations of existing methods, some (largely the pruning methods) were created by me.

--ActivationFunctions.py--
Contains all activation functions which I wrote from scratch. This includes SERLU, Maxout,
and Channel-out (itself with two versions, max and absolute max). It also contains a 
function for generating a module list of activation function layers, given a function name 
and any relevent arguments. Some of these are functions I did not write from scratch and
only call from a Torch library, but I included them nontheless for comparison and ease of use.

--DropoutLayers.py--
Identical in function and form to ActivationFunctions, the only difference being that I define 
dropout classes here, and have a function which creates a list of layers of them given relevent 
arguments.

--PruningNet.py--
Contains a definition for a network with a variety of activation functions, normalization layers, and regularization layers, along with class functions for various methods of pruning. There is also a function for training this network given certain parameters.
