import csv
import tarfile
import io
import datetime
from typing import NamedTuple, OrderedDict, BinaryIO
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

CSV_FILENAME = 'vanderbilt.csv'
PICKLE_FILENAME = 'vanderbilt.pickle'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

class WeatherRow(NamedTuple):
    id: int
    time: str
    derived: int
    barometer: float
    barometer_tendency: str
    thermometer: float
    dewpoint: float
    heat_index: float
    wet_bulb_globe_temperature: float
    wind_chill: float
    anemometer: int
    ten_minute_wind_gust: int
    hygrometer: int
    wind_vane: int
    solar_radiation: int
    uv_radiation: int
    rain_rate: float
    rain_gauge: float
    turf_temperature: str
    club_level_temperature: str
    club_level_humidity: str
    field_level_temperature: str
    field_level_humidity: str

    @classmethod
    def from_csv_dict(cls: 'WeatherRow', dct: OrderedDict[str, str]) -> dict:
        """ Convert a weather dict read with DictReader to correct types as WeatherRow."""
        new_dict = {}
        for field, value in dct.items():
            try:
                new_dict[field] = cls.__annotations__[field](value)
            except:
                print(f'Failed to cast for id={dct["id"]}: {field} {value} {cls.__annotations__[field]}')
                return None
        return new_dict

def read_from_tar(filename: str) -> BinaryIO:
    tar = tarfile.open(f'{filename}.tar.xz')
    return (tar, tar.extractfile(tar.getmember(filename)))

if __name__ == '__main__':
    if not os.path.isfile(f'{PICKLE_FILENAME}.tar.xz'):
        data_tar, data_binary = read_from_tar(CSV_FILENAME)
        data_str = io.TextIOWrapper(data_binary, encoding='utf-8')
        data_reader = csv.DictReader(data_str)
        weather = [WeatherRow.from_csv_dict(dct) for dct in data_reader]
        data_tar.close()
        with tarfile.open(f'{PICKLE_FILENAME}.tar.xz', 'w:xz') as pickle_tar:
            pickle_bytes = pickle.dumps(weather, pickle.HIGHEST_PROTOCOL)
            pickle_tarinfo = tarfile.TarInfo(PICKLE_FILENAME)
            pickle_tarinfo.size = len(pickle_bytes)
            pickle_fobj = io.BytesIO(pickle_bytes)
            pickle_tar.addfile(pickle_tarinfo, fileobj=pickle_fobj)

    pickle_tar, pickle_binary = read_from_tar(PICKLE_FILENAME)
    weather = pickle.load(pickle_binary)
    pickle_tar.close()
    print(weather[0])
