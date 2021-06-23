
## Audio to Face Blendshape 
Implementation with PyTorch. 

复现人：刘宇昂
- Base model
    - LSTM using MFCC audio features
    - CNN([ref](http://research.nvidia.com/publication/2017-07_Audio-Driven-Facial-Animation) simplified version) with LPC features


## Prerequisites
- Python3
- PyTorch v0.3.0
- numpy
- librosa & audiolazy
- scipy
- etc.

## Files
- Scripts to run
  - `main.py`: change net name and set checkpoints folder to train different models
  - `test_model.py`: generate blendshape sequences given extracted audio features (need audio features as input)
  - `synthesis.py`: generate blendshape directly from input wav (need arguements of input audio path)

- Classes
  - `models.py`: Classes with LSTM and CNN (simplified NvidiaNet) model. 
  - `models_testae.py`: Advanced models with audoencoder design. 
  - `dataset.py`: Class for loading dataset.

- Input preprocessing
  - `misc/audio_mfcc.py`: extract mfcc features from input wav files
  - `misc/audio_lpc.py`: extract lpc features
  - `misc/combine.py`: combine certain audio feature/blendshape files to obtain a single file for data loading

## Usage
### Input
To build your own dataset, you need to preprocess your wav/blendshape pairs with `misc/audio_mfcc.py` or `misc/audio_lpc.py`. Then combine those feature/blendshape files `misc/combine.py` to a single feature/blendshape file. 

### Training
Modify `main.py`. Set model to the one you need and also specify checkpoint folder. 

### Evaluation
- Both `test_model.py` and `synthesis.py` can be used to generate blendshape sequences. 
    - `test_model.py` accepts extrated audio features (MFCC/LPC).
    - `synthesis.py` takes raw wav file as input
    - State the arguments and it will produce a blenshape test file. 


