# CAVE on reconstruction of Chinese written characters

This repository contains the code and instructions for working with a dataset and training a Variational Autoencoder (VAE) model. Follow the steps below to get started.

## Getting Started

### Step 1: Download the Dataset

Run the following command to download the dataset:
```bash get_dataset.sh```

### Step 2: Extract Images

Run the following function exactly once to transform the downloaded .gnt files to .png files:
```extract_images(gnt_dir, output_dir)```
Replace gnt_dir with the path to your .gnt files and output_dir with the path where you want to save the .png files.

### Step 3: Explore the Data and Training Process

Open and walk through the data.ipynb notebook to get an idea of what each image looks like and understand the training process of the VAE model.



