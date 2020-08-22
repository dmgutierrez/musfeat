from .analysis.audio_analysis import AudioProcessor
from .models.audio_models import AudioAnalysisFeature, NaiveFeaturesDim
from .helper.settings import logger


class MusFeat(object):
    def __init__(self, sr: int = 44100, n_mels: int = 2**5, hop_length: int = 512, n_fft: int = 2048,
                 fmin: int = 50, roll_percent_min: float = 0.1, roll_percent_max: float = 0.85,
                 roll_percent: float = 0.5, power: float = 2.0, n_chroma: int = 12,
                 n_octaves: int = 7, mono: bool = True):

        self.audio_processor: AudioProcessor = AudioProcessor(
            sr=sr, n_mels=n_mels, hop_length=hop_length,
            n_fft=n_fft, fmin=fmin,
            roll_percent_min=roll_percent_min,
            roll_percent_max=roll_percent_max,
            roll_percent=roll_percent, power=power,
            n_chroma=n_chroma, n_octaves=n_octaves,
            mono=mono)

    def get_naive_audio_features_from_file(self, audio_file_path: str, window: bool = True,
                                           seconds_per_analysis: int = 6):
        audio_features: AudioAnalysisFeature = object.__new__(AudioAnalysisFeature)
        try:
            audio_features: AudioAnalysisFeature = self.audio_processor.execute_audio_analysis(
                audio_file_path=audio_file_path,
                window=window,
                seconds_per_analysis=seconds_per_analysis)
        except Exception as e:
            logger.error(e)
        return audio_features

    def get_naive_features_dim(self, seconds_per_analysis: int = 6):
        naive_feature_dim: NaiveFeaturesDim = object.__new__(NaiveFeaturesDim)
        try:
            # 1. Get length
            feature_length: int = self.audio_processor.get_feature_length(
                sr=self.audio_processor.sr,
                hop_length=self.audio_processor.hop_length,
                seconds=seconds_per_analysis)

            naive_feature_dim.mel: tuple = (self.audio_processor.n_mels,
                                            feature_length)
            naive_feature_dim.chroma: tuple = (self.audio_processor.n_chroma,
                                               feature_length)
        except Exception as e:
            logger.error(e)
        return naive_feature_dim
