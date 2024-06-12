# Chinese Written Character Reconstruction Project

This repository contains code and models for reconstructing Chinese written characters using various Conditional Variational Autoencoder (CVAE) architectures. The project focuses on loading and preprocessing data, training different models, and conducting ablation studies to compare their performance.

## Getting Started

### Downloading the Dataset
1. Download https://github.com/skishore/makemeahanzi/tree/master/svgs
2. Run `convert_data.py` and store them in `./chinese_chars/pngs`

### Models

All model architectures are stored under the `./models` folder.

### Testing the Model

To get an initial idea of how to test our model, open and run the `CVAE_Half.ipynb` notebook. This will guide you through the testing process.

### Training the Model

To train a model for each architecture:

1. Open the corresponding `.ipynb` file (e.g., `CVAE.ipynb`).
2. Run the code in the notebook to train the model.
3. After training, save the model under the `./saves` directory for further analysis and ablation studies.

### Ablation Study

To compare the results of different models, run the `ablation_study.ipynb` notebook. This will provide a detailed comparison of the performance of the various trained models.

## Directory Structure

- `./chinese_chars/pngs` - Directory for storing the dataset.
- `./model` - Directory containing all the model architectures.
- `./saves` - Directory for storing the trained models.
- `CVAE_Half.ipynb` - Notebook for initial taste of the model.
- `CVAE.ipynb` - Notebook for training vanilla CVAE.
- `ablation_study.ipynb` - Notebook for conducting the ablation study to compare model performance.

## Contributing

Feel free to submit issues or pull requests if you have any improvements or bug fixes. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
