[![DOI](https://zenodo.org/badge/696941025.svg)](https://zenodo.org/doi/10.5281/zenodo.11287438)

# Predictive Coding Neural Network for Constructing Cognitive Maps

Welcome to the official GitHub repository accompanying our latest research paper. Our work dives deep into the cognitive processes humans employ to understand their surroundings and positions themselves in space. This repository contains the implementation and details of the novel framework we've introduced, leveraging the versatility of predictive coding in constructing spatial maps.

## Overview

Humans have an innate ability to generate cognitive maps of their environments from direct sensory inputs, even without explicit coordinates or measurements. In the domain of machine learning, while approaches like SLAM capture visual features and construct spatial maps using visual data, they may not capture the holistic nature of cognitive maps in the human brain. Our study is focused on understanding and replicating this generalized mapping strategy that can adapt to various sensory data including visual, auditory, tactile, and linguistic inputs.

## Key Features

- **Predictive Coding Neural Network**: A foundational aspect of our work, this neural network algorithm demonstrates the potential for constructing spatial maps using a variety of sensory data.
  
- **Virtual Environment Navigation**: Within our framework, we have an agent that navigates a virtual environment, employing visual predictive coding via a convolutional neural network equipped with self-attention.

- **Next Image Prediction**: As our agent learns this task, it subsequently constructs an internal representation of the environment, reflecting the inherent spatial distances.

- **Vectorized Encoding**: The predictive coding network produces a vectorized representation of the environment. This not only allows vector navigation but also emphasizes the concept of localized, overlapping neighborhoods in the virtual space.

## Implications

Broadly speaking, our work not only offers insights into human cognitive processes but also introduces predictive coding as a unified algorithmic strategy. The potential applications and extensions to auditory, sensorimotor, and linguistic mappings make this a versatile tool for various research and practical applications.

## Getting Started

This repository contains the codebase, sample environments, and the datasets used for our experiments. Please refer to the detailed documentation for installation instructions, usage guidelines, and insights into our methodology.

Thank you for your interest in our work. We eagerly await the community's feedback, collaborations, and future contributions to this exciting domain.

### Notebooks
`environment.ipynb`: [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgornet/predictive-coding-recovers-maps/blob/main/notebooks/environment.ipynb): 
In this Google Colab notebook, we will be setting up the Malmo framework, a platform designed to harness the rich environment of Minecraft for research in artificial intelligence.

`train.ipynb`: [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgornet/predictive-coding-recovers-maps/blob/main/notebooks/train.ipynb)
In this Google Colab notebook, we detail the procedure for training a predictive coding and autoencoding neural network using a dataset derived from a Minecraft environment.

`predictive_coding.ipynb`: [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgornet/predictive-coding-recovers-maps/blob/main/notebooks/predictive_coding.ipynb)
In this Google Colab notebook, we apply a pre-trained predictive coding neural network to a dataset containing observations from an agent navigating the Minecraft environment.

`autoencoding.ipynb`: [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgornet/predictive-coding-recovers-maps/blob/main/notebooks/autoencoding.ipynb)
In this Google Colab notebook, we apply a pre-trained autoencoding neural network to a dataset containing observations from an agent navigating the Minecraft environment.

`circular.ipynb`: [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgornet/predictive-coding-recovers-maps/blob/main/notebooks/circular.ipynb)
In this Google Colab notebook, we demonstrate that the predictive coder can learn a circular topology and distinguish visually separate, spatially different locations.

`vector_navigation.ipynb`: [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jgornet/predictive-coding-recovers-maps/blob/main/notebooks/vector_navigation.ipynb)
In this Google Colab notebook, we will analyze the latent units of the predictive coding neural network to demonstrate how it performs vector navigation.

---

For further details, kindly refer to our paper. If you have queries or suggestions, please open an issue or reach out to us directly.
