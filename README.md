# Musfeat: A simplified Python wrapper to compute Music features

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

A Python library for computing different music features for deep learning purposes using Librosa.

# New Features!

  - Given an audio filepath, generate naive audio features including the so-called Mel Spectrogram as well as the Chromagram
  - Window analysis for computing local features 
  - Python Iterators for large scale scenarios
  - Mel Spectogram is normalized using the so-called [Per-Channel Energy Normalization (PCEN)](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf)

### 1. Installation

You can easily install the module via pip.

```sh
$ pip install musfeat
```


### 2. How to use it
##### Generate Music features in a straightforward fashion
In this first example, we are going to collect both Mel Spectrogram and Chromagram using a window analysis with a window size of 6 seconds. Thus, the module generates the aforementioned features for each non-overlapped temporal window and it yields an iterator with the corresponding data. Let's dive in with a simple example:


```python
import os
from musfeat.musfeat import MusFeat
from musfeat.models.audio_models import AudioAnalysisFeature, NaiveFeaturesDim

# Audio Path Parameter
audio_file_path: str = os.path.join("samples", "sample_02.mp3")

# Temporal window analysis (True in this case)
window: bool = True
# Seconds of the analysis (6 seconds in this example)
seconds_per_analysis: int = 6

# Create the default MusFeat Object (You must select the audio parameters you desired)
musfeat_obj: MusFeat = MusFeat()

# Call the naive function which returns both Mel and Chromagram
audio_features: AudioAnalysisFeature = musfeat_obj.get_naive_audio_features_from_file(
    audio_file_path=audio_file_path, window=window,
    seconds_per_analysis=seconds_per_analysis)
```

##### Check the dimension of your features in a very simple way

Image you have generated all your audio features and you have them in a database. If you want to build a Deep Learning model using them, you will need to pass to the model the dimension of the desired features. Since you already know the parameters you used for your analysis, there is no need to recover one sample of your database in order to check the dimension. You can just call this method:

```python
# Get feature dims
feature_dims: NaiveFeaturesDim = musfeat_obj.get_naive_features_dim(
    seconds_per_analysis=seconds_per_analysis)
```

The obtained object, will contain the dimensions of your features.

### Data Models
Let's see the main attributes of the entities involved in the package.

#### MusFeat Data Model
Attribute | Type | Description
------------ | ------------- | ------------- 
sr | integer | sample rate
n_mels | integer | number of mel filters
hop_length | integer | number of samples between consecutive audio frames
n_fft | integer | number of points in the fft
fmin | integer | minimum frequency (maximum frequency is computed using Nyquist)
roll_percent_min | float | minimum roll percentage (spectral analysis)
roll_percent_max | float | maximum roll percentage  (spectral analysis)
roll_percent | float | roll percentage  (spectral analysis)
power | float | power
n_chroma | integer | Number of chroma bins to produce
n_octaves | integer | Number of octaves to analyze above fmin
mono | boolean | indicator whether you want to work with mono or non-mono signals during the analysis

For more information, you can consult the official Librosa library where these parameters are very well described: [Librosa](https://librosa.org/doc/latest/index.html). 

#### AudioAnalysisFeature Data Model
Attribute | Type | Description
------------ | ------------- | ------------- 
mel | iterator | data related to the Mel Spectrogram computation
chroma | iterator | data related to the Chromagram computation
mel_size | tuple | shape of the Mel feature
chroma_size | tuple | shape of the Chromagram feature
n_segments | integer | total number of segments obtained during the analysis

The n_segments feature indicates how many features you have obtained after performing your window analysis. As expected, **if you not perform a temporal window analysis**, you will only have one feature so that **n_segments will be equal to 1**.

#### NaiveFeaturesDim Data Model

Attribute | Type | Description
------------ | ------------- | ------------- 
mel | tuple | shape of the Mel feature
chroma | tuple | shape of the Chromagram feature

For more information, you can consult the official Spotify Web API regarding Artists at [Get an Artist](https://developer.spotify.com/documentation/web-api/reference/artists/get-artist/). 

### *TODOs*

 - Write more documentation within the code
 - Include more music features

License
----

MIT License

Copyright (c) 2020 DAVID MARTIN-GUTIERREZ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the MusFeat software), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Contact

For further explanations and doubts do not hesitate to send an email to dmargutierrez@gmail.com.