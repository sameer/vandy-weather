import io
import csv
import tarfile
import os
import pickle
from typing import BinaryIO, List, Tuple

import torch
import numpy as np

from weather_format import WeatherRow, BarometerTendency

CSV_FILENAME = 'vanderbilt.csv'
PICKLE_FILENAME = 'vanderbilt.pickle'
TORCH_FILENAME = 'vanderbilt.pt'


def read_from_tar(filename: str) -> Tuple[tarfile.TarFile, BinaryIO]:
    tar = tarfile.open(f'{filename}.tar.xz')
    return (tar, tar.extractfile(tar.getmember(filename)))


if __name__ == '__main__':
    if not os.path.isfile(f'{PICKLE_FILENAME}.tar.xz'):
        print('Loading data')
        data_tar, data_binary = read_from_tar(CSV_FILENAME)
        data_str = io.TextIOWrapper(data_binary, encoding='utf-8')
        data_reader = csv.DictReader(data_str)
        print('Processing data')
        weather = [WeatherRow.from_csv_dict(dct) for dct in data_reader]
        weather = list(filter(lambda w: w is not None, weather))
        data_tar.close()
        print('Saving pickle (compressing)')
        with tarfile.open(f'{PICKLE_FILENAME}.tar.xz', 'w:xz') as pickle_tar:
            pickle_bytes = pickle.dumps(weather, pickle.HIGHEST_PROTOCOL)
            pickle_tarinfo = tarfile.TarInfo(PICKLE_FILENAME)
            pickle_tarinfo.size = len(pickle_bytes)
            pickle_fobj = io.BytesIO(pickle_bytes)
            pickle_tar.addfile(pickle_tarinfo, fileobj=pickle_fobj)

    if not os.path.isfile(f'{TORCH_FILENAME}.tar.xz'):
        print('Loading pickle')
        pickle_tar, pickle_binary = read_from_tar(PICKLE_FILENAME)
        weather: List[WeatherRow] = pickle.load(pickle_binary)
        pickle_tar.close()
        print('Converting to PyTorch 2D array')
        data = np.empty((len(weather), len(WeatherRow._fields)))
        for i, w in enumerate(weather):
            for j in range(len(WeatherRow._fields)):
                if j == 1:
                    data[i, j] = w[j].timestamp()
                elif j == 4:
                    data[i, j] = w[j].value
                elif j >= 18:  # Bad sensor values
                    data[i, j] = 0
                else:
                    data[i, j] = w[j]
        print('Saving PyTorch 2D array (compressing)')
        with tarfile.open(f'{TORCH_FILENAME}.tar.xz', 'w:xz') as torch_tar:
            buffer = io.BytesIO()
            torch.save(data, buffer)
            torch_tarinfo = tarfile.TarInfo(TORCH_FILENAME)
            torch_tarinfo.size = buffer.seek(0, io.SEEK_END)
            print(buffer.tell())
            torch_tar.addfile(torch_tarinfo,
                              fileobj=io.BytesIO(buffer.getvalue()))
    print('Done!')
