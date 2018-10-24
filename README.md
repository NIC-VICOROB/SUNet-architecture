# SUNet-architecture

Development framework for evaluation of deep learning architectures for "SUNet: a deep learning architecture for acute stroke lesion segmentation and outcome prediction in multimodal MRI".

## Installation

The method makes use of Keras and Tensorflow. If the method is running on GPU, please make sure CUDA and `tensorflow-gpu` are correctly installed. 

```
pip install -r requirements.txt
```

## Running the code

1. Read [ISLES challenge registration instructions](https://www.smir.ch/ISLES/Start2017) in the 'How to join' section and register.

2. Download and extract the [ISLES2015](https://www.smir.ch/ISLES/Start2015) (SISS and SPES) and [ISLES2017](https://www.smir.ch/ISLES/Start2017) datasets.

3. Update the path to each dataset in the file `configuration.py` (line 137).

4. Load the three pre-trained SUNet models and reproduce the cross-validation results in the paper by running 
`python main.py`.

5. Interpreting the results:
   * The resulting probability maps and binary segmentations can be found in each dataset folder inside the `results` folder.
   * The evaluation metrics csv can be found in the `metrics` folder.
