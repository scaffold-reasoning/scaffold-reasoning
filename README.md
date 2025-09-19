# Scaffold Reasoning Implementation

<!-- [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](link-to-paper) -->

This repository contains the **official implementation** of **Scaffold Reasoning** as described in our paper:  
**"Dual-Process Scaffold Reasoning for Enhancing LLM Code Debugging"**  
*Authors: po-chung hsieh, chin-po chen, jeng-lin lee, ming-ching chang*  
<!-- (Conference/Journal, Year)   -->

## Overview

Scaffold Reasoning is a psychologically inspired framework for LLM code debugging, combining Scaffold, Analytic, and Integration streams. It structures intermediate reasoning (System 2) to guide final outputs (System 1), improving both accuracy and efficiency. Achieves 88.91% pass rate on DebugBench with 5.36s per problem.


## Key Features

- **Scaffold Stream**: Generates reference code to guide the debugging process.  
- **Analytic Stream**: Analyzes buggy code to identify errors and potential fixes.  
- **Integration Stream**: Merges insights from Scaffold and Analytic streams to produce accurate debugging solutions.  
- **Main results from the original paper**: Achieves 88.91% pass rate on DebugBench with 5.36s per problem.  



## Installation

### Requirements

- transformers >= 4.17.0.dev0 – Core library for loading and running LLMs.

- tiktoken >= 0.5.0 – Tokenization for LLM inputs and outputs.

- datasets >= 1.8.0 – Loading DebugBench and other benchmark datasets.

- accelerate – Efficient model inference and distributed computation.

- numpy < 2.0 – Numerical computations and array handling in analysis and integration.

### Quick Install

```bash
git clone https://github.com/scaffold-reasoning/scaffold-reasoning.git
cd scaffold-reasoning
pip install -r requirements.txt
```


## Usage

### Quick Start

```python
bash run_modelXprompt.sh --max-questions [NUMBER]
```



## Paper Results

This implementation reproduces all results reported in our paper. All experiments were conducted using the code in this repository.

### Main Results
Pass Rates of reasoning methods on DebugBench (%). **Bold** indicates the best result.

| Method | GPT-4o | GPT-4.1-mini | Devstral-Small-1.1 | CodeQwen2.5-32B | Avg. Pass Rate (%)↑ | AvgPTime (s)↓ |
|--------|--------|--------------|-------------------|-----------------|-------------------|---------------|
| Base | 84.33 | 85.56 | 67.80 | 84.04 | 80.43 | 5.62 |
| CoT | 84.22 | 85.21 | 67.61 | 83.80 | 80.21 | 6.93 |
| Pearl | 85.25 | 84.59 | 66.61 | 82.16 | 79.65 | 7.17 |
| ReAct | 85.65 | 86.40 | 67.90 | 83.69 | 80.91 | 5.49 |
| LDB | 86.97 | 86.43 | 67.81 | 84.79 | 81.50 | 5.47 |
| CoA | 84.97 | 83.50 | 67.27 | 77.91 | 78.41 | 6.36 |
| **SR (Ours)** | **87.23** | **88.91** | **69.52** | **85.69** | **82.84** | **5.36** |


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<!-- ## Citation

If you use this implementation in your research, please cite both the original paper and this repository:

```bibtex
@article{author2024framework,
  title={Paper Title},
  author={Author Names},
  journal={Journal/Conference Name},
  year={2024},
  url={link-to-paper}
}

@software{your2024implementation,
  title={Framework Name Implementation},
  author={Your Name},
  url={https://github.com/username/repo-name},
  year={2024}
}
``` -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original paper authors for their excellent work
- Any other acknowledgments
- Contributors to this implementation

## Contact

- **Maintainer**: po-chung hsieh (pochung.hsieh@gmail.com)
- **Issues**: Please report bugs and feature requests through [GitHub Issues](link-to-issues)



<!-- if running Wizard code model, remember to use pytorch higher than 2.6



cuda message
---
Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-12.6/

Please make sure that
 -   PATH includes /usr/local/cuda-12.6/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.6/lib64, or, add /usr/local/cuda-12.6/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.6/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 560.00 is required for CUDA 12.6 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
--- -->
