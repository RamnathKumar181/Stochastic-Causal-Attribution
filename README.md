# Stochastic-Causal-Attribution

The code repository for the project - Stochastic Insight into Causal Attribution. This work was performed during my internship at AmazonML and published at Amazon Machine Learning Conference (AMLC) 2021.
The major contributions of this work are as follows:

* We propose a robust unsupervised method for Stochastic Causal Attribution. Our system exploits networks such as deep quantile networks or mixture density networks to capture the stochasticity of the model. To the best of our knowledge, this is the first attempt to study stochastic causal attribution, specifically in the regression setting.
* Unlike previous approaches, the stochastic approach allows us to consider the causal errors that might originate in a few-shot setting.
* The definition of a stochastic causal attribution allows us to compare two different causal attributions, not only by their mean but also their uncertainty.
* We prove empirically that mixture density networks are better capable of understanding the problem and causal attributions than traditional neural networks, which are trained with a quantile loss.
* We validate our proposed methodology on both simulated datasets and real datasets. Our model consistently performs well on all problems.

In this repo, we share our codes from our experiment settings for simulated datasets as well as real datasets (except AmazonDF dataset, since the ownership of this dataset remains with Amazon). The codes have been simplified and have been segregated based on the dataset, for easy use and understanding. The original paper published at AMLC2021 and AmazonDF dataset cannot be made public since AmazonML has proprietary rights to the work and data. The code published here is only for the public datasets and is not a violation of the ethics of AmazonML.

# Setup
### Requirements

Listed in `requirements.txt`. Install with `pip install -r
requirements.txt` preferably in a virtualenv.

### Data
All the simulated datasets have been created manually, and do not need any additional steps from your side. Furthermore, the real datasets including boston dataset and diabetes dataset have been used using `sklearn.datasets` and will be downloaded upon running the script! :)
