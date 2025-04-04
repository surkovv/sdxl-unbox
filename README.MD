# Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2410.22366) 
[![Hugging Face Spaces Demo](https://img.shields.io/badge/Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/surokpro2/Unboxing_SDXL_with_SAEs) 
[![Colab](https://img.shields.io/badge/Colab-Notebook-yellow)](https://colab.research.google.com/drive/1Sd-g3w2Fwv7pc_fxgeQOR3S_RKr18qMP?usp=sharing)


![modification demostration](resourses/image.png)

This repository contains code to reproduce results from our paper on using sparse autoencoders (SAEs) to analyze and interpret the internal representations of text-to-image diffusion models, specifically SDXL Turbo.


## Repository Structure

```
|-- SAE/                    # Core sparse autoencoder implementation
|-- SDLens/                 # Tools for analyzing diffusion models
|   `-- hooked_sd_pipeline.py   # Modified stable diffusion pipeline
|-- scripts/
|   |-- collect_latents_dataset.py  # Generate training data
|   `-- train_sae.py                    # Train SAE models
|-- utils/
|   `-- hooks.py           # Hook utility functions
|-- checkpoints/           # Pretrained SAE model checkpoints
|-- app.py                # Demo application
|-- app.ipynb             # Interactive notebook demo
|-- example.ipynb         # Usage examples
`-- requirements.txt      # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Demo Application

You can try our gradio demo application (`app.ipynb`) to browse and experiment with 20K+ features of our trained SAEs out-of-the-box. You can find the same notebook on [Google Colab](https://colab.research.google.com/drive/1lWZ2yCRwCf4iuykvb-91QYUNkuzIwI3k?usp=sharing).

## Usage

1. Collect latent data from SDXL Turbo:
```bash
python scripts/collect_latents_dataset.py --save_path={your_save_path}
```

2. Train sparse autoencoders:

    2.1. Insert the path of stored latents and directory to store checkpoints in `SAE/config.json`

    2.2. Run the training script:

```bash
python scripts/train_sae.py
```

## Pretrained Models

We provide pretrained SAE checkpoints for 4 key transformer blocks in SDXL Turbo's U-Net in the `checkpoints` folder. See `example.ipynb` for analysis examples and visualization of learned features. More pretrained SAEs with different parameters are accessible through [HuggingFace repo](https://huggingface.co/surokpro2/sdxl-saes/tree/main).


## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@misc{surkov2024unpackingsdxlturbointerpreting,
      title={Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders}, 
      author={Viacheslav Surkov and Chris Wendler and Mikhail Terekhov and Justin Deschenaux and Robert West and Caglar Gulcehre},
      year={2024},
      eprint={2410.22366},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.22366}, 
}
```

## Acknowledgements

The SAE component was implemented based on [`openai/sparse_autoencoder`](https://github.com/openai/sparse_autoencoder) repository.

