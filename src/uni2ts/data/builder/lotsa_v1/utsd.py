from collections import defaultdict
from functools import partial
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder


class UTSDDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "AtrialFibrillation_data",
        "AustraliaRainfall",
        "BeijingPM25Quality",
        "BenzeneConcentration",
        "BIDMC32HR",
        "EigenWorms",
        "IEEEPPG",
        "MotorImagery",
        "Phoneme",
        "PigArtPressure",
        "PigCVP",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "StarLightCurves",
        "TDBrain",
        "UTSD_SENSORDATA",
        "Worms",
    ]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(self, *args, **kwargs):
        # Has been pre-built
        pass
