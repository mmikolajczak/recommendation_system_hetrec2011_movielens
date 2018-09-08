import os
import os.path as osp
import pandas as pd
from recommendations_system.io_ import load_hetrec_to_df
from recommendations_system.io_._utils import flatten, is_iterable


HETREC_DATA_PATH = '../../data/hetrec2011-movielens-2k-v2'
OUTPUT_VOCABS_PATH = '../../data/generated_categories_vocabs'


def generate_categories_vocab_file(series, output_path, encoding='utf-8'):
    if is_iterable(series.iloc[0]) and type(series.iloc[0]) != str:
        series = pd.Series(flatten(series.tolist()))
    unique = pd.unique(series).tolist()

    with open(output_path, 'w', encoding=encoding) as f:
        # vocab_file.txt should contain one line for each vocabulary element (src: docs)
        for val in unique:
            f.write(f'{val}\n')
        # # Note: NaN - added especially to simplify work with encoders in generator
        # f.write('NaN\n')


def generate_hetrec_vocab_files(data_path, output_path, encoding='utf-8'):
    os.makedirs(output_path, exist_ok=True)
    hetrec_df = load_hetrec_to_df(data_path)

    features_series = (hetrec_df['userID'], hetrec_df['movieID'], hetrec_df['actorID'], hetrec_df['country'],
                       hetrec_df['directorID'], hetrec_df['genre'], hetrec_df['location'])
    out_files_names = tuple(f'{feature_name}_vocab.txt' for feature_name in ('users', 'movies', 'actors', 'countries',
                                                                             'directors', 'genres', 'locations'))

    for series, out_file_name in zip(features_series, out_files_names):
        generate_categories_vocab_file(series, osp.join(output_path, out_file_name), encoding=encoding)


if __name__ == '__main__':
    generate_hetrec_vocab_files(HETREC_DATA_PATH, OUTPUT_VOCABS_PATH)

