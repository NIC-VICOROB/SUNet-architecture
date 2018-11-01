# SUNet: a deep learning architecture for acute stroke lesion segmentation and outcome prediction in multimodal MRI

Development framework for evaluation of deep learning architectures in the paper (https://arxiv.org/abs/1810.13304)

## Installation

The method makes use of Keras and Tensorflow. If the method is running on GPU, please make sure CUDA 9.X is correctly installed. Then, in the base directory run: 
```
pip install -r requirements.txt
```

## Running the code

1. Read [ISLES challenge registration instructions](https://www.smir.ch/ISLES/Start2017) in the 'How to join' section and register.

2. Download and extract the [ISLES2015](https://www.smir.ch/ISLES/Start2015) (SISS and SPES) and [ISLES2017](https://www.smir.ch/ISLES/Start2017) datasets.

3. Update the dataset dictionary with the path to each dataset in `configuration.py` (line 137).

4. Reproduce the cross-validation results in the paper by running :

   ```
   python main.py
   ```

   For each performed cross-validation:
      1. The included pre-trained models from `checkpoints/` will be loaded for the corresponding fold. 
   
      2. The corresponding validation images of the training set will be segmented. 
   
      3. Finally, the computed evaluation metrics will be written to a spreadsheet file.

5. Accessing the results:
   + The resulting binary segmentations will be found in the `results/` folder.
   + A spreadsheet with the evaluation metrics for each crossvalidation will be in the `metrics/` folder.
