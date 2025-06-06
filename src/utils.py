# -*- coding: utf-8 -*-

import json
import os
import pickle
from collections import defaultdict, OrderedDict, Counter
from itertools import product
import numpy as np
import pandas as pd
from read_config import *
from scipy.stats import truncnorm
from surprise import SVD, Dataset, Reader

# from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def predict_consumers_items_utilities(
    ratings_df: pd.DataFrame,
) -> (defaultdict, object):
    """

    :param ratings_df: pd.DataFrame of consumers' ratings.
    :return: dict to store the items' predictions for each consumer and an object of the trained recommender engine model.

     A function computes consumers' utilities of all the available items using a machine learning algorithm.
    """
    print("Predicting consumers items' utilities, it will take sometime...")
    data_path = get_dataset_dir()
    items_path = os.path.join(data_path, model_input["items_dataset"])
    items = pd.read_csv(items_path)["movieId"].unique()
    users = ratings_df["userId"].unique()
    reader = Reader(rating_scale=(0.5, 5))
    ratings_df = ratings_df.iloc[:, :3]
    data = Dataset.load_from_df(ratings_df, reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    predictions = defaultdict(list)

    for uid in tqdm(users):
        for iid in items:
            est = model.predict(uid, iid).est
            predictions[uid].append({"iid": int(iid), "rating": float(round(est, 3))})
    print("End predicting utilities")
    return predictions, model


def get_ordered_recs(predictions: defaultdict) -> defaultdict:
    """

    :param ratings_df: pd.DataFrame of consumers' ratings.
    :return: defaultdict to store the items' predictions for each consumer.

    A function takes the items' predictions and sorts them in descending order.
    """
    for uid, user_ratings in predictions.items():
        user_ratings.sort(key=lambda x: x["rating"], reverse=True)
        predictions[uid] = user_ratings
    return predictions


def get_ratings_data() -> pd.DataFrame:
    """

    :return: pd.DataFrame of the ratings dataset.

     A function reads the consumers' ratings of previous watch movies from the MovieLens
     (https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) dataset.
    """
    datasetdir = get_dataset_dir()
    ratings_data_path = os.path.join(datasetdir, model_input["ratings_dataset"])
    ratings_df = pd.read_csv(ratings_data_path)
    return ratings_df


def store_recommender_systems_generated_data() -> None:
    """

    A function stores the data generated by the recommender systems algorithm into pickle files.
     It is executed only once to store the predictions for further experimentations.
    """
    recdata_path = get_rec_dir()
    create_directory(recdata_path)
    if len(os.listdir(recdata_path)) < 3:
        ratings_df = get_ratings_data()
        predictions, model = predict_consumers_items_utilities(ratings_df)
        popular_items = get_popular_items(ratings_df)

        if not os.path.exists(f"{recdata_path}/SVDmodel.p"):
            pickle.dump(
                model,
                open(f"{recdata_path}/SVDmodel.p", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if not os.path.exists(
            f"{recdata_path}/consumers_items_utilities_predictions.p"
        ):
            recommendations = get_ordered_recs(predictions)
            pickle.dump(
                recommendations,
                open(f"{recdata_path}/consumers_items_utilities_predictions.p", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        if not os.path.exists(
            f"{recdata_path}/consumers_items_utilities_predictions_popular.p"
        ):
            num_cons = len(list(ratings_df["userId"].unique()))
            recommendations = get_predictions_popular_items(
                num_cons, popular_items, model
            )
            pickle.dump(
                recommendations,
                open(
                    f"{recdata_path}/consumers_items_utilities_predictions_popular.p",
                    "wb",
                ),
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def get_top_n(predictions: defaultdict, n: int) -> defaultdict:
    """

    :param predictions: defaultdict of consumers and their proposed recommendations.
    :param n: int of number of items in the recommended list.
    :return: defaultdict of items with the `n` highest predicted ratings for each consumer.

    A function takes the items of the highest ratings as recommendations.
    """
    topn_recommendations = defaultdict(list)
    for uid, recs in predictions.items():
        topn_recommendations[uid] = recs[:n]
    return topn_recommendations


def get_predictions_popular_items(
    num_consumers: int, popular_items: list, svdmodel: object
) -> defaultdict:
    """
    :param num_consumers: int of the number of consumers .
    :param popular_items: list of the popular items.
    :param svdmodel: object of the svd trained model
    :return: defaultdict to store the items' predictions for each consumer.

    A function adds the predicted utilities of the popular items.
    """
    predictions = defaultdict(list)
    for uid in range(1, num_consumers + 1):
        for iid in popular_items:
            est = svdmodel.predict(uid, iid).est
            predictions[uid].append({"iid": int(iid), "rating": float(round(est, 3))})
    return predictions


def get_popular_items(ratings_df: pd.DataFrame) -> OrderedDict:
    """
    :param ratings_df: pd.DataFrame of the ratings data.
    :return: list of popular items.

    A function finds the popular items in the dataset. The most popular items are the ones that have the most ratings from the given dataset.
    """

    rated_items = list(ratings_df["movieId"])
    popular_items = dict(Counter(rated_items))
    for item in get_items():
        popular_items.get(item, 0)

    popular_items = dict(
        OrderedDict(sorted(popular_items.items(), key=lambda i: i[1], reverse=True))
    )
    return popular_items


def rescale_rating(consumer_true_utility: float) -> float:
    """

    :param consumer_true_utility: float of consumer's true utility.
    :return: float of consumer's true utilities scaled.

    A function maps a given `true utility` to a given scale
    """
    scales = model_input["rating_scale"]
    n = len(scales)
    if consumer_true_utility in scales or consumer_true_utility == 0:
        pass
    elif consumer_true_utility < scales[0]:
        consumer_true_utility = scales[0]
    elif consumer_true_utility > scales[n - 1]:
        consumer_true_utility = scales[n - 1]
    else:
        for i in range(1, n):
            if consumer_true_utility < scales[i]:
                mid = (scales[i - 1] + scales[i]) / 2
                if consumer_true_utility < mid:
                    consumer_true_utility = scales[i - 1]
                else:
                    consumer_true_utility = scales[i]
                break
    return consumer_true_utility


def rerank_per_consumer(
    items_with_ratings_predictions: list, profits: dict, weights: list
) -> list:
    """

    :param items_with_ratings_predictions: list of the items objects predictions for all consumers.
    :param profits: dict of profit data given to each item.
    :param weights: list of two float values in range [0,1].
    :return: list of dicts of items.

    A function reranks items based on a weighted sum based on predicted utilities of items for each consumer and profit of one consumer.
    """
    # model_parameters["balanced_strategy"]["RW"]  # weight for the rating
    consumer_w = weights[0]
    provider_w = weights[1]
    for i in items_with_ratings_predictions:
        i["rank"] = consumer_w * i["rating"] + provider_w * profits[i["iid"]]
        i["profit"] = profits[i["iid"]]
    items_with_ratings_predictions.sort(key=lambda x: x["rank"], reverse=True)
    return items_with_ratings_predictions


def rerank_items_consider_profit(
    items_with_ratings_predictions: defaultdict, profits: dict, weights: list
) -> defaultdict:
    """

    :param items_with_ratings_predictions: defaultdict of items predictions for all consumers.
    :param profits: dict of profit data given to each item.
    :param weights: list of two float values in range [0,1]
    :return: defaultdict of new items ranking.

    A function uses `rerank_per_consumer` function to reorder items by balancing consumer utility and profit
    """
    ordered_items_for_recommendations = defaultdict(list)
    for uid, recs in items_with_ratings_predictions.items():
        ordered_items_for_recommendations[uid] = rerank_per_consumer(
            recs, profits, weights
        )
    return ordered_items_for_recommendations


def get_num_items() -> int:
    """

    :return: int.

    A function to get the number of consumers in the dataset
    """
    datasetdir = get_dataset_dir()
    items_data_path = os.path.join(datasetdir, model_input["items_dataset"])
    items_df = pd.read_csv(items_data_path)
    return items_df["movieId"].nunique()


def get_items() -> int:
    """

    :return: list[int].

    A function to get items ids.
    """
    datasetdir = get_dataset_dir()
    items_data_path = os.path.join(datasetdir, model_input["items_dataset"])
    items_df = pd.read_csv(items_data_path)
    return list(items_df["movieId"].unique())


def create_directory(dir: str) -> None:
    """

    A function creates a directory by giving a path.
    """
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            print(f"An issue while creating the directory: {dir}")
        else:
            print(f"Successfully created the directory {dir}")


def get_data_dir() -> str:
    """

    :return: str.
    A function gets the absolute path of the data directory.
    """
    current_dir = os.path.join(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, model_input["data_directory"])
    return data_path


def get_dataset_dir() -> str:
    """

    :return: str.
    A function gets the absolute path of the dataset directory.
    """
    data_dir = get_data_dir()
    dataset_path = os.path.join(data_dir, model_input["dataset_directory"])
    return dataset_path


def get_results_dir() -> str:
    """

    :return: int.
    A function gets the absolute path of the results directory.
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    results_path = os.path.join(parent_dir, model_input["results_directory"])
    return results_path


def get_rec_dir() -> str:
    """
    :return: str.
    A function gets the absolute path of the directory of the recommendations data
    """
    data_dir = get_data_dir()
    recdata_path = os.path.join(data_dir, model_input["recommendation_data_directory"])
    return recdata_path


def get_exec_path() -> str:
    """

    :return: str.
    A function gets the absolute path of the current execution directory.
    """
    result_path = get_results_dir()
    exec_path = os.path.join(result_path, model_input["execution_dir"])
    return exec_path


def get_sensitive_params() -> list:
    """

    :return: list.
    A function sets the sensitive parameters based on experiments, we added the most sensitive parameters.
     sensitive parameters: strategy, expectation_threshold_quantile, error.
    """
    sensitive_params = ["recommendation_strategy", "quantile_consumer_expectation"]
    return sensitive_params


def get_params(obj: object) -> str:
    """

    :param obj: object of agents or model class.
    :return: str represents a senario name.

    A function gets the values of sensitive model parameters and uses the generated dictionary of parameters as keys to distinguished scenarios.
    """
    params = get_sensitive_params()
    d_params = {}
    for p in params:
        if obj.__class__.__name__ == "ConsumerAgent":
            d_params[p] = getattr(obj.model, p)
        else:
            d_params[p] = getattr(obj, p)
    return SCENARIOS[str(d_params)]


def generate_profitdata(seed: float, items: list) -> dict:
    """

    :param obj: An object of agents or model class.
    :return: str represents a senario name.

     A function generates random profit data for each item drawn from normal distribution (mu=2.5,sigma=1).
    different seed values are used each simulation iteration.


    """
    np.random.seed(seed)
    # items = get_items()
    lower, upper = 0, 5
    mu, sigma = 2.5, 1
    profits = truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
    ).rvs(len(items))
    items_profits = {i: round(p, 2) for i, p in zip(items, profits)}
    return items_profits


def create_scenarios(sensitive_params: list) -> dict:
    """

    :param: A list of parameters.
    :return: dict of scenarios.

    A function generates senarios based on the defined sensitive parameters.

    """
    results_path = get_results_dir()
    create_directory(results_path)
    params_values = [model_parameters[p] for p in sensitive_params]
    products_vals = list(product(*params_values))
    SCENARIOS = {}
    for i, p in enumerate(products_vals):
        k = {}
        for j, v in enumerate(p):
            k[sensitive_params[j]] = v
        SCENARIOS[str(k)] = "scenario" + str(i + 1)
    return SCENARIOS


sensitive_params = get_sensitive_params()
SCENARIOS = create_scenarios(sensitive_params)


def store_scenarios(path: str) -> None:
    """

    A function stores the output scenarios in json file.

    """
    with open(f"{path}/SCENARIOS.json", "w") as fp:
        json.dump(SCENARIOS, fp, indent=4)


def update_consumer_personal_experiences(exp: float) -> float:
    """
    :param exp: float of the distance between the true utility of the consumed item and the expectation threshold.
    A function updates consumer's personal experience, three ditance functions can be used "euclidean", "manhattan", "binary".

    """
    distance = model_parameters["trust_update_distance"]
    if distance == "euclidean":
        inc = exp * exp
    # elif distance == "manhattan":
    #     inc = d
    else:
        inc = 1
    return inc


def generate_profit_data_popularity(seed: int, items) -> dict:
    """

    :param seed: a value to generate random values.
    A function to generate profit data that is positively correlated with items popularity.
    """

    profit_data = generate_profitdata(seed, items)
    ratings_df = get_ratings_data()
    popular_items = get_popular_items(ratings_df)

    profit_data = dict(sorted(profit_data.items(), key=lambda x: x[1], reverse=True))
    pop_sorted = list(popular_items.values())

    profit_pop_corr_data = {}

    i = 0
    for iid, pop in popular_items.items():
        profit = list(profit_data.values())[i]
        if pop_sorted[i] == pop_sorted[i - 1]:
            profit = list(profit_data.values())[i - 1]
        profit_pop_corr_data[iid] = profit
        i += 1

    return profit_pop_corr_data


def compute_trust_parameters(ratings: pd.DataFrame, m: int) -> list:
    """

    :param ratings (pd.DataFrame): Contains user-items and ratings.
    :param m (int): The mean of ratings.

    :returns: A list of two values a and b.
    A function to count the number of items based on a condition.
    """
    a = len([i for i in ratings if i >= m])
    b = len([i for i in ratings if i < m])
    return [a, b]


def initialize_beta(df) -> None:
    """

    :param df (pd.Dataframe): Contains user-items and ratings.

    :return: None.
    The method initializes the parameters
    of beta distribution that are used to compute the consumer trust.
    """
    recdata_path = get_data_dir()
    initials_betapath = os.path.join(
        recdata_path, model_input["trust_data_dir"], "beta_initials.p"
    )
    m = np.median(df["rating"])
    beta_initials = {}
    consumed_items = defaultdict(list)
    groups = df.groupby("userId")
    n = df.userId.nunique()
    for i in range(1, n + 1):
        g = groups.get_group(i)
        beta_initials[i] = compute_trust_parameters(list(g["rating"]), m)
        consumed_items[i].append(list(g["movieId"]))

    pickle.dump(
        beta_initials, open(initials_betapath, "wb"), protocol=pickle.HIGHEST_PROTOCOL
    )
