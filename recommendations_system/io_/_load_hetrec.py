import os.path as osp
import numpy as np
import pandas as pd


def load_hetrec_to_df(data_dir_path, encoding='utf-8'):
    """
    Loads hetrec data into pandas DataFrame. Currently only minimal subset of data, containing info only about users
    movie rating (identified by pair user_id/movie_id) is being loaded.
    :param data_dir_path: path to the directory containing unzipped hetrec data.
    :param encoding: encoding o files in data directory. Default is utf-8, but note that hetrec data can be downloaded
    from Internet also in some ISO encoding.
    :return: pandas df containing data about particular review. All data about reviewed movie (location, actors, etc...)
    is aggregated in each row. Tags are not used (at least for now)
    TODO: check if it can be null in data, or it only contains the actually made reviews, and also memory consumption in this approach
    TODO: possible column data types adjustments.
    """
    rated_movies_file_path = osp.join(data_dir_path, 'user_ratedmovies.dat')
    rated_movies_df = pd.read_csv(rated_movies_file_path, sep='\t', usecols=['userID', 'movieID', 'rating'],
                                  encoding=encoding,
                                  dtype={'userID': np.uint64, 'movieID': np.uint64, 'rating': np.float32})

    movies_actors_file_path = osp.join(data_dir_path, 'movie_actors.dat')
    movies_actors_df = pd.read_csv(movies_actors_file_path, sep='\t', usecols=['movieID', 'actorID'], encoding=encoding)
    # movies_actors_df.rename({'ranking': 'actorRanking'}, inplace=True) we ignore actors ranking for now
    movies_actors_df = movies_actors_df.groupby('movieID').agg(lambda x: x.tolist())
    hetrec_combined = rated_movies_df.join(movies_actors_df, on='movieID')
    actor_nulls_cond = pd.isnull(hetrec_combined['actorID'])
    hetrec_combined.loc[actor_nulls_cond, 'actorID'] = [[[]] * actor_nulls_cond.sum()]

    movies_countries_file_path = osp.join(data_dir_path, 'movie_countries.dat')
    movies_countries_df = pd.read_csv(movies_countries_file_path, sep='\t', encoding=encoding)
    hetrec_combined = pd.merge(hetrec_combined, movies_countries_df, on='movieID', how='left')
    hetrec_combined.loc[pd.isnull(hetrec_combined['country']), 'country'] = ''

    movies_directors_file_path = osp.join(data_dir_path, 'movie_directors.dat')
    movies_directors_df = pd.read_csv(movies_directors_file_path, sep='\t', usecols=['movieID', 'directorID'],
                                      encoding=encoding)
    hetrec_combined = pd.merge(hetrec_combined, movies_directors_df, on='movieID', how='left')
    hetrec_combined.loc[pd.isnull(hetrec_combined['directorID']), 'directorID'] = ''

    movies_genres_file_path = osp.join(data_dir_path, 'movie_genres.dat')
    movies_genres_df = pd.read_csv(movies_genres_file_path, sep='\t', encoding=encoding)
    movies_genres_df = movies_genres_df.groupby('movieID').agg(lambda x: x.tolist())
    hetrec_combined = hetrec_combined.join(movies_genres_df, on='movieID')

    movies_locations_file_path = osp.join(data_dir_path, 'movie_locations.dat')
    movies_locations_df = pd.read_csv(movies_locations_file_path, sep='\t', usecols=['movieID', 'location1',
                                                                                      'location2'], encoding=encoding)
    movies_locations_df['location'] = movies_locations_df['location1'] + ', ' + movies_locations_df['location2']
    movies_locations_df.drop(['location1', 'location2'], axis=1, inplace=True)
    movies_locations_df = movies_locations_df.groupby('movieID').agg(lambda x: x.tolist())
    hetrec_combined = hetrec_combined.join(movies_locations_df, on='movieID')

    assert len(hetrec_combined) == len(rated_movies_df)
    return hetrec_combined