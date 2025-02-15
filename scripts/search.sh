#!/bin/sh

python beam_predict.py

# To ensure reproducibility of results in paper in spite of stochasticity, 
# we aggregate results from a 100 random seeds
# python examples/beam_predict_seed.py