################################################################################
                       # HYBRID RECOMMENDER SYSTEM #
################################################################################

# Make an estimate for the user whose ID is given, using the item-based and user-based recommender methods.
# Consider 5 suggestions from the user-based model and 5 suggestions
# from the item-based model and finally make 10 suggestions from 2 models.
# user_id = 108170

################################################################################
                          # Data Preprocessing #
################################################################################

import pandas as pd

pd.set_option ('display.max_columns', 20)
pd.set_option ('display.width', None)

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()

rare_movies = comment_counts[comment_counts["title"] <= 1000].index  # to filter data
rare_movies[0:10]


def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("datasets/movie.csv")
    rating = pd.read_csv("datasets/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

user_movie_df.head()
user_movie_df.shape

################################################################################
    # Determining the Movies Watched by the User to Make a Suggestion #
################################################################################

random_user = 108170
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()

random_user_df.notna().any()
random_user_df.columns[random_user_df.notna().any()]
type(random_user_df.columns[random_user_df.notna().any()])  #pandas.core.indexes.base.Index
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched[0:10]

################################################################################
    # Data and ID information of Other Users Watching the Same Movies #
################################################################################

len(movies_watched)
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.T.head()

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count.head()

user_movie_count = user_movie_count.reset_index()
user_movie_count.head()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.sort_values(by="movie_count", ascending=False)

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies.head()

################################################################################
       # Determining the Users to be Suggested and Most Similar Users #
################################################################################

movies_watched_df.head()
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies.values)]
final_df.head()
final_df.shape

final_df[final_df.index == random_user]

final_df.T.head()
final_df.T.corr().head()
final_df.T.corr().unstack().head()
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.head()

corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values (by='corr', ascending=False)
top_users.head()

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.head()

################################################################################
 # Calculating Weighted Average Recommendation Score and Keeping Top 5 Movies #
################################################################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df.head()
recommendation_df = recommendation_df.reset_index()


import matplotlib.pyplot as plt
recommendation_df["weighted_rating"].hist()
plt.show()

# weighted_rating greater than 2.7
recommendation_df[recommendation_df["weighted_rating"] > 2.7]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 2.7].\
    sort_values("weighted_rating", ascending = False)
movies_to_be_recommend.head()

movie = pd.read_csv ('datasets/movie.csv')
recommended_user_based_df = movies_to_be_recommend.merge (movie[["movieId", "title"]])
recommended_user_based_df.head()
recommended_user_based_df.shape

# Suggesting movies that the user hasn't watched before
recommended_user_based_df = recommended_user_based_df.loc[~recommended_user_based_df["title"].isin(movies_watched)][:5]

#    movieId  weighted_rating                       title
# 0     8376         3.582032    Napoleon Dynamite (2004)
# 1     1449         3.582032  Waiting for Guffman (1996)
# 2     2804         3.582032   Christmas Story, A (1983)
# 3     2195         3.582032           Dirty Work (1998)
# 4     6188         3.582032           Old School (2003)

################################################################################
                        # Item-Based Recommendation
################################################################################

# Make an item-based suggestion based on the name of the movie that the user has watched with the highest score.
# Make 10 suggestions with 5 suggestions user-based and 5 suggestions item-based.

# Clue:

# user = 108170

# movie = pd.read_csv('datasets/movie.csv')
# rating = pd.read_csv('datasets/rating.csv')

# Obtaining the id of the movie with the most recent score from the movies
# that the user to be recommended gives 5 points:
# movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
# sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]


user = 108170

movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/rating.csv')


# Getting the id of the movie with the most recent score from the movies that
# the user to be suggested gives 5 points:
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)]. \
               sort_values (by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie.loc[movie["movieId"] == movie_id, "title"]

user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
user_movie_df.corrwith(movie).sort_values(ascending=False).head(5)


def item_based_recommender(movie_name, user_movie_df, head=10):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith (movie).sort_values(ascending=False).head(head)


movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].
                                                values[0], user_movie_df, 20).reset_index()
movies_from_item_based.head()
movies_from_item_based.rename(columns={0:"corr"}, inplace=True)
movies_from_item_based.head()

# Suggesting movies that the user has not watched before
recommended_item_based_df = movies_from_item_based.loc[~movies_from_item_based["title"].isin(movies_watched)][:5]
recommended_item_based_df

#                                    title      corr
# 1              My Science Project (1985)  0.570187
# 2                    Mediterraneo (1991)  0.538868
# 3        Old Man and the Sea, The (1958)  0.536192
# 4  National Lampoon's Senior Trip (1995)  0.533029
# 5                   Clockwatchers (1997)  0.483337

hybrid_rec_df = pd.concat([recommended_user_based_df["title"], recommended_item_based_df["title"]]).reset_index(drop=True)
hybrid_rec_df


