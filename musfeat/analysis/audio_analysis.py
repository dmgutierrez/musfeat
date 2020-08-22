# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:51:19 2019

@author: dmarg
"""

import librosa as li
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub.utils import mediainfo
from ..helper.settings import logger
from ..models.audio_models import NaiveAudioFeature, AudioAnalysisFeature


class AudioProcessor(object):
    def __init__(self, sr: int = 44100, n_mels: int = 2 ** 5, hop_length: int = 512, n_fft: int = 2048,
                 fmin: int = 50, roll_percent_min: float = 0.1, roll_percent_max: float = 0.85,
                 roll_percent: float = 0.5, power: float = 2.0, n_chroma: int = 12,
                 n_octaves: int = 7, mono: bool = True):
        self.sr: int = sr
        self.n_mels: int = n_mels
        self.hop_length: int = hop_length
        self.n_fft: int = n_fft
        self.fmin: int = fmin
        self.fmax: int = int(sr / 2)
        self.roll_percent_min: float = roll_percent_min
        self.roll_percent_max: float = roll_percent_max
        self.roll_percent: float = roll_percent
        self.power: float = power
        self.n_chroma: int = n_chroma
        self.n_octaves: int = n_octaves
        self.mono: bool = mono

    def execute_audio_analysis(self, audio_file_path: str, window: bool = True,
                               seconds_per_analysis: int = 5):
        audio_features: AudioAnalysisFeature = object.__new__(AudioAnalysisFeature)
        try:
            # 1. Load audioFile
            logger.info(f"Loading File {audio_file_path}")

            y, sr = self.load_audio(file_path=audio_file_path, sr=self.sr, mono=self.mono)
            mel_data: list = []
            chroma_data: list = []
            segment_id: int = 1
            naive_feature_segment: NaiveAudioFeature = object.__new__(NaiveAudioFeature)

            # 2.1 Window Analysis
            if window:
                logger.info(f"Generating features using a temporal window of {seconds_per_analysis} (s)")
                n_samples: int = int(seconds_per_analysis * self.sr)
                done: bool = True
                offset: int = 0
                while done:
                    if offset + n_samples > len(y):
                        done = False
                    else:
                        y_analysis: np.array = y[offset: offset + n_samples]
                        offset += n_samples
                        # Process features
                        naive_feature_segment: NaiveAudioFeature = self.compute_naive_audio_features(
                            y=y_analysis)
                        # Add segment id to naive feature object
                        mel_data.append(naive_feature_segment.mel.flatten().tolist())
                        chroma_data.append(naive_feature_segment.chroma.flatten().tolist())
                        segment_id += 1

            # 2.2 Non-windows Analysis
            else:
                logger.info(f"Generating features without using a temporal window")
                # Process features
                naive_feature_segment: NaiveAudioFeature = self.compute_naive_audio_features(
                    y=y)
                mel_data.append(naive_feature_segment.mel.flatten().tolist())
                chroma_data.append(naive_feature_segment.chroma.flatten().tolist())

            # 3. Add Original sizes
            audio_features: AudioAnalysisFeature = AudioAnalysisFeature(
                mel_data=mel_data,
                chroma_data=chroma_data,
                mel_size=naive_feature_segment.mel_size,
                chroma_size=naive_feature_segment.chroma_size,
                segment_id=segment_id)
        except Exception as e:
            logger.error(e)
        return audio_features

    def compute_naive_audio_features(self, y: np.array):
        naive_feature: NaiveAudioFeature = object.__new__(NaiveAudioFeature)
        try:
            feature_length: int = self.extract_feature_length_from_file(y, sr=self.sr, n_fft=self.n_fft,
                                                                        hop_length=self.hop_length)
            # Compute Mel spectrogram
            mel: np.ndarray = __class__.compute_mel_spectrogram(
                y, sr=self.sr, feature_length=feature_length,
                n_mels=self.n_mels, power=self.power,
                n_fft=self.n_fft, hop_length=self.hop_length,
                fmin=self.fmin, fmax=self.fmax)
            naive_feature.mel: np.ndarray = __class__.normalize_mel_spectrogram(s=mel, power=self.power)
            naive_feature.mel_size: tuple = (self.n_mels, feature_length)

            # Compute Chromagram-cqt
            naive_feature.chroma: np.ndarray = __class__.compute_chromagram_cqt(
                y=y, sr=self.sr, feature_length=feature_length,
                n_chroma=self.n_chroma, n_octaves=self.n_octaves,
                hop_length=self.hop_length)
            naive_feature.chroma_size: tuple = (self.n_chroma, feature_length)
        except Exception as e:
            logger.error(e)
        return naive_feature

    def extract_feature_length_from_file(self, y: np.array, sr: int, n_fft: int, hop_length: int):
        feature_length: int = -1
        try:
            seconds: int = self.__class__.get_duration(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            feature_length: int = self.__class__.get_feature_length(seconds=seconds,
                                                                    sr=sr, hop_length=hop_length)
        except Exception as e:
            logger.error(e)
        return feature_length

    @staticmethod
    def get_sample_rate(file_path: str):
        sr: int = 44100
        try:
            audio_info: dict = mediainfo(file_path)
            sr: int = int(audio_info['sample_rate'])
        except Exception as e:
            logger.error(e)
        return sr

    @staticmethod
    def load_audio(file_path: str, sr: int = None, mono: bool = True):
        y: np.array = None
        try:
            y, sr = li.load(file_path, sr=sr, mono=mono)
        except Exception as e:
            logger.error(e)
        return y, sr

    @staticmethod
    def get_duration(y: np.array, sr: int, n_fft: int, hop_length: int):
        seconds: int = -1
        try:
            seconds: int = li.core.get_duration(y, sr, n_fft=n_fft, hop_length=hop_length)
        except Exception as e:
            logger.error(e)
        return seconds

    @staticmethod
    def normalize_mel_spectrogram(s: np.array, power: float):
        s_db: np.array = np.array([])
        try:
            if power == 2.0:
                s_db = li.power_to_db(s, ref=np.max)
            elif power == 1.0:
                s_db = li.amplitude_to_db(s, ref=np.max)
            else:
                # Per-Channel Energy normalization
                s_db = librosa.pcen(s)
        except Exception as e:
            logger.error(e)
        return s_db

    @staticmethod
    def get_feature_length(seconds: int, sr: int, hop_length: int):
        feature_length: int = -1
        try:
            feature_length: int = int(np.ceil((seconds*sr) / hop_length))
        except Exception as e:
            logger.error(e)
        return feature_length

    @ staticmethod
    def compute_mfcc(y: np.array, sr: int, feature_length: int, n_mels: int = 64,
                     n_fft: int = 2048, hop_length: int = 512):
        # Init value
        mfcc = -99 * np.ones((n_mels, feature_length))
        try:
            mfcc = li.feature.mfcc(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        except Exception as e:
            logger.error(e)
        return mfcc

    @staticmethod
    def compute_chromagram_cqt(y: np.array, sr: int, feature_length: int, n_chroma: int = 12,
                               n_octaves: int = 7, hop_length: int = 512):
        # Init value
        chroma_cqt = -99 * np.ones((n_chroma, feature_length))
        try:
            chroma_cqt = li.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma,
                                               n_octaves=n_octaves)
        except Exception as e:
            logger.error(e)
        return chroma_cqt

    @staticmethod
    def compute_mel_spectrogram(y: np.array, sr: int, feature_length: int, n_mels: int = 64,
                                power: float = 2.0, n_fft: int = 2048, hop_length: int = 512,
                                fmin: int = 20, fmax: int = 8000):
        # Init value
        mel: np.array = -99 * np.ones((n_mels, feature_length))
        try:
            mel: np.array = li.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                      power=power, n_mels=n_mels, fmin=fmin, fmax=fmax)
        except Exception as e:
            logger.error(e)
        return mel

    @staticmethod
    def compute_zero_crossing_rate(y: np.array, feature_length: int, frame_length: int = 2048,
                                   hop_length: int = 512):
        # Init value
        zcr: np.array = -99 * np.ones((1, feature_length))
        try:
            zcr: np.array = li.feature.zero_crossing_rate(y=y, frame_length=frame_length,
                                                          hop_length=hop_length,
                                                          center=True)
        except Exception as e:
            logger.error(e)

        return zcr

    @staticmethod
    def compute_tonnetz(y: np.array, sr: int, feature_length: int, chroma_cqt: np.array = None):
        # Init value
        tonnetz: np.array = -99 * np.ones((6, feature_length))
        try:
            tonnetz: np.array = li.feature.tonnetz(y=y, sr=sr, chroma=chroma_cqt)
        except Exception as e:
            logger.error(e)
        return tonnetz

    @staticmethod
    def plot_mel_spectrogram(s: np.array, sr: int, fmin: int, fmax: int, hop_length: int):
        try:
            plt.figure()
            librosa.display.specshow(s, x_axis='time',
                                     y_axis='mel', sr=sr,
                                     fmax=fmax, fmin=fmin, hop_length=hop_length)

            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-frequency spectrogram')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(e)

    @staticmethod
    def plot_chroma_cqt_spectrogram(s: np.array, sr: int, hop_length: int):
        try:
            plt.figure()
            librosa.display.specshow(s, x_axis='time',
                                     y_axis='chroma', sr=sr, hop_length=hop_length)
            plt.colorbar()
            plt.title('Chromagram CQT')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(e)