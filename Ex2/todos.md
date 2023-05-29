TODOs must:
- 2d activation function (sigmoid) - David
- hyperparameter optimization - Lexi
- experiments on datasets 
    - breast Alina
    - fertility - Alina
    - loan
- preprocessing of categorical data inside algorithm - Alina
- batch processing for bigger dataset? 
- slides - everyone

TODOs bonus:
- regularization 
- dropout

Aussagen die wir aussagen wollen:
- 5 models: ours, best NN, default NN, best other, default other
    - Performance
    - Runtime

- Comparison of 2 activation functions:
    - Performance - Done
    - Runtime - Done
    - Training vs validation loss sigmoid vs relu - for best models only?
    - Convergence random init weights vs optimized init weights
    - Vanishing gradient problems - model converges, but accuracy is under threshold (<10-20%)
    - Differences in learning rates

- Network structure
    - 5 nodes 1 layer still holds in comparison to more complex structures
    - more complex dataset still needs more complex structure
    - Convergence of models becomes a bigger problem with higher model complexity
    - Training vs validation loss - small vs big model, small vs big dataset

TODO: export plots as svg