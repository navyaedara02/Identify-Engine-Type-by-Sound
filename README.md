# Identify Engine Type by Sound

This project uses signal processing techniques and convolutional neural network (CNN) to identify the number of cylinders for an internal combustion engine from the sound.

## File Description

| file | description |
| --- | --- |
| [engine_id.ipynb](./engine_id.ipynb) | Main notebook containing PyTorch CNN model |
| [signal_processing.py](./signal_processing.py) |  Signal processing library |
| [signal_processing_demo.ipynb](./signal_processing_demo.ipynb) | Demostrating signal processing techniques used in [signal_processing.py](./signal_processing.py) |

## Environment

```sh
# Create environment
python3 -m venv .venv

# Initiate environment (Unix-like OS's)
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```
