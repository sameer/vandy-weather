from enum import Enum, auto
from typing import NamedTuple, Dict
import datetime

import torch
from torch.utils.data import Dataset
from sklearn import preprocessing

from config import WINDOW_SIZE

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

class BarometerTendency(Enum):
    NULL = auto()
    Falling_Rapidly = auto()
    Falling_Slowly = auto()
    Steady = auto()
    Rising_Slowly = auto()
    Rising_Rapidly = auto()


class WeatherRow(NamedTuple):
    '''
    Id: SQL Table Key, int
    Time: Timestamp, datetime.datetime
    Derived: ????
    Barometer: Inches of Mercury in. Hg
    Barometer Tendency: Pressure Tendency  
    Thermometer: Degrees Fahrenheit &deg;F
    Dewpoint: Degrees Fahrenheit &deg;F
    Heat Index: Degrees Fahrenheit &deg;F
    Wet Bulb Globe Temperature: Degrees Fahrenheit &deg;F
    Wind Chill: Degrees Fahrenheit &deg;F
    Anemometer: Miles Per Hour mph
    10 Minute Wind Gust: Miles Per Hour mph
    Hygrometer: Percent Humidity %
    Wind Vane: Degrees &deg;
    Solar Radiation Sensor: Watts Per Square Meter W/m^2
    UV Radiation Sensor: UV Index  
    Rain Rate: Inches Per Hour in/hr
    Rain Gauge: Inches in.
    Turf Temperature: Degrees Fahrenheit &deg;F
    Club Level Temperature: Degrees Fahrenheit &deg;F
    Club Level Humidity: Percent Humidity %
    Field Level Temperature: Degrees Fahrenheit &deg;F
    Field Level Humidity: Percent Humidity %
    '''
    id: int
    time: datetime.datetime
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
    def from_csv_dict(cls: 'WeatherRow', dct: Dict[str, str]) -> dict:
        """ Convert a weather dict read with DictReader to correct types as WeatherRow."""
        new_dict = {}
        for field, value in dct.items():
            try:
                if field == 'time':
                    new_dict[field] = datetime.datetime.strptime(
                        value, TIME_FORMAT)
                elif field == 'barometer_tendency':
                    new_dict[field] = BarometerTendency[value.replace(
                        ' ', '_')]
                else:
                    new_dict[field] = cls.__annotations__[field](value)
            except:
                print(
                    f'Failed to cast for id={dct["id"]}: {field} {value} {cls.__annotations__[field]}'
                )
                return None
        return cls(**new_dict)


class WeatherDataset(Dataset):
    def __init__(self, data):
        self.scaler = preprocessing.StandardScaler()
        self.data = torch.from_numpy(self.scaler.fit_transform(data.cpu().numpy().reshape(-1, 1)).reshape(-1)).to(data.device, dtype=data.dtype)

    def __getitem__(self, idx: int):
        return self.data[idx:idx+WINDOW_SIZE]


    def __len__(self):
        return len(self.data) - WINDOW_SIZE
