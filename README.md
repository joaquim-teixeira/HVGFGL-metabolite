# Hierarchical Variational Graph Fused Lasso for Recovering Relative Rates in Spatial Compositional Data

This repository contains the implementation code for the paper:

**"A Hierarchical Variational Graph Fused Lasso for Recovering Relative Rates in Spatial Compositional Data"**  
*arXiv:2509.20636*

## Repository Structure

```
├── notebooks/
│   ├── HVGFGL_Final_Model_Implementations.ipynb    # HVGFL model implementation on simulation, along with simpler models. Includes plots and summary statistics
│   ├── HVGFGL_Kidney_Data_VI.ipynb  # HVGFL model implementation on Mouse Kidney Data
│   ├── Plot_and_SSIM_Kidney_Data.ipynb    # Plots kidney data 
├── data/
│   └── kidney_tic/                    # Real IMS dataset from Wang, et al. 2022 (https://pubmed.ncbi.nlm.nih.gov/35132243/)
├── requirements.txt                   # Python dependencies
├── utils.py                           # Includes all custom functions
└── README.md                          # This file
```



### Running on Google Colab

All notebooks are designed to run on Google Colab. Simply:
1. Open the notebook in Colab
2. Install dependencies by running the first cell
3. Follow the step-by-step instructions in each notebook


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{teixeira2024hierarchical,
  title={A Hierarchical Variational Graph Fused Lasso for Recovering Relative Rates in Spatial Compositional Data},
  author={Teixeira, Joaquim, Reznik, Ed, Banerjee, Sudipto, and Tansey, Wesley,
  journal={arXiv preprint arXiv:2509.20636},
  year={2025}
}
```

## Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest improvements or new features
- Submit pull requests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaborations, please contact:
Joaquim Teixeira - joaquimteixeira@ucla.edu
