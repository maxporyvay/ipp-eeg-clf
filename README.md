# Internally pronounced phonemes EEG classifier

Run with
```
python3 -m classifier
```

Edit classifier configuration in classifier/config.py

EDF files are translated to spectrograms with Morlet wavelet transform. Here are some examples of the resulting spectrograms:

![Phoneme: A; EEG Channel: T3; Person: Antonova](https://raw.githubusercontent.com/maxporyvay/ipp-eeg-clf/main/datasets/morlet-examples/Phoneme_A_Channel_T3_Antonova.jpeg)
![Phoneme: F; EEG Channel: F7; Person: Antonova](https://raw.githubusercontent.com/maxporyvay/ipp-eeg-clf/main/datasets/morlet-examples/Phoneme_F_Channel_F7_Antonova.jpeg)

Then CNN classifier is used to classify the spectrograms
