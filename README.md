## Content
Generative model for producing simple sentences with multiple possible syntactic structures, and model inversion using the variational Bayesian method.

`DEM_MDP_agent_bias_var1.m` defines the generative model. The input to the model `sen` is a numeric vector, each entry indexing a word in the dictionary stored in `Knowledge_AB.mat`.

`spm_MDP_VB_X_agent_bias.m` implements the variational Bayesian model inversion
