import numpy as np


class NaiveAudioFeature(object):
    def __init__(self, mel: np.ndarray, chroma: np.ndarray,
                 mel_size: tuple, chroma_size: tuple):
        self.mel: np.ndarray = mel
        self.chroma: np.ndarray = chroma
        self.mel_size: tuple = mel_size
        self.chroma_size: tuple = chroma_size


class AudioAnalysisFeature(object):
    def __init__(self, mel_data: list, chroma_data: list,
                 mel_size: tuple, chroma_size: tuple, segment_id: int):
        self.mel: iter = iter(mel_data)
        self.chroma: iter = iter(chroma_data)
        self.mel_size: tuple = mel_size
        self.chroma_size: tuple = chroma_size
        self.n_segments: int = segment_id - 1


class NaiveFeaturesDim(object):
    def __init__(self, mel_size: tuple, chroma_size: tuple):
        self.mel: tuple = mel_size
        self.chroma: tuple = chroma_size