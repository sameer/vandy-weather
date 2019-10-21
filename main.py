import csv
import tarfile
import io

FILENAME = 'vanderbilt.csv'
data_tar = tarfile.open(f'{FILENAME}.tar.xz')
data_tar_member = data_tar.getmember(FILENAME)
data_bytes = data_tar.extractfile(data_tar_member)
data = io.TextIOWrapper(data_bytes, encoding='utf-8')

data_reader = csv.DictReader(data)
print(next(data_reader))
