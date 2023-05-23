# Internally pronounced phonemes EEG classifier

Run with
```
python3 -m classifier
```

Edit classifier configuration in classifier/config.py

EDF files are translated to spectrograms with Morlet wavelet transform. Here are some examples of the resulting spectrograms:

![Phoneme: A; EEG Channel: T3; Persson: Antonova](https://github.com/maxporyvay/ipp-clf-eeg/blob/main/datasets/morlet-examples/Phoneme_A_Channel_T3_Antonova.jpeg?raw=true)
![Phoneme: F; EEG Channel: F7; Persson: Antonova](https://github.com/maxporyvay/ipp-clf-eeg/blob/main/datasets/morlet-examples/Phoneme_F_Channel_F7_Antonova.jpeg?raw=true)

Then CNN classifier is used to classify the spectrograms
