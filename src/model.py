# -*- coding: utf-8 -*-

import itertools
import pickle
import time
from collections import defaultdict
import numpy as np
import random
from consumer import ConsumerAgent
from mesa.model import Model
from mesa_utils.datacollection import DataCollector
from mesa_utils.schedule import RandomActivationByType


from read_config import *
from service_provider import ProviderAgent
from utils import *


class RecommendationModel(Model):
    c = itertools.count(1)

    def __init__(self, **kwargs):
        super().__init__(self)
        self.schedule = RandomActivationByType(self)
        self.running = True

        self.initialize_parameters(**kwargs)
        # Setup agents
        self.create_provider()
        self.create_consumers()

        # collecting data from the simulation
        self.datacollector = DataCollector(
            model_reporters={
                "strategy": "recommendation_strategy",
                "switch_to_strategy": lambda m: (m.switch_to_strategy),
                "model_params": get_params,
                "step": lambda m: m.schedule.steps,
                "total_profit": lambda m: np.round(m.total_profit, 3),
                "number_of_consumption": lambda m: np.round(m.number_of_consumption, 3),
                "avg_profit_per_consumption": lambda m: np.round(
                    m.avg_profit_per_consumption, 3
                ),
                "number_of_dropout": lambda m: len(m.dropout_consumers),
                # "number_of_positive_posts": lambda m: np.round(np.sum([p[0] for p in m.social_media[: self.schedule.steps]])),
                # "number_of_negative_posts": lambda m: np.round(
                #     np.sum([p[1] for p in m.social_media[: self.schedule.steps]])
                # ),
                "social_reputation": lambda m: np.round(m.social_reputation, 3),
                "social_reputation_moving_avg": lambda m: (
                    np.round(
                        np.mean(
                            m.social_reputation_history[
                                m.schedule.steps
                                - model_parameters["n"] : m.schedule.steps
                                + 1
                            ]
                        ),
                        3,
                    )
                    if m.schedule.steps > model_parameters["n"]
                    else np.round(m.social_reputation, 3)
                ),
                "profit_moving_avg": lambda m: (
                    np.round(
                        np.mean(
                            m.total_profit_history[
                                m.schedule.steps
                                - model_parameters["n"] : m.schedule.steps
                                + 1
                            ]
                        ),
                        3,
                    )
                    if m.schedule.steps > 10
                    else np.round(m.total_profit, 3)
                ),
            },
            agent_reporters={
                "step": lambda a: (
                    a.model.schedule.steps
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "consumerId": lambda a: (
                    a.consumer_id if a.__class__.__name__ == "ConsumerAgent" else None
                ),
                "trust": lambda a: (
                    a.trust if a.__class__.__name__ == "ConsumerAgent" else None
                ),
                "minimum_utility_threshold": lambda a: (
                    a.minimum_utility_threshold
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "true_utility": lambda a: (
                    a.true_utility if a.__class__.__name__ == "ConsumerAgent" else None
                ),
                "selecteditem": lambda a: (
                    a.selecteditem if a.__class__.__name__ == "ConsumerAgent" else None
                ),
                "consumption_probability": lambda a: (
                    a.consumption_probability
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "lower_limit": lambda a: (
                    a.consumption_probability_limits[0]
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "upper_limit": lambda a: (
                    a.consumption_probability_limits[1]
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "model_params": lambda a: (
                    get_params(a) if a.__class__.__name__ == "ConsumerAgent" else None
                ),
                "strategy": lambda a: (
                    a.model.recommendation_strategy
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "is_satisfied": lambda a: (
                    a.is_satisfied()
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "num_positive_personal": lambda a: (
                    np.round(a.positive_negative_experience[0], 3)
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
                "num_negative_personal": lambda a: (
                    np.round(a.positive_negative_experience[1], 3)
                    if a.__class__.__name__ == "ConsumerAgent"
                    else None
                ),
            },
        )
        self.datacollector.collect(self)

    def add_new_items(self):
        print("New items are added to the item catalog......")
        sampled_items = random.sample(self.new_items, model_parameters["num_items"])

        item_profit = generate_profitdata(self.seed, sampled_items)

        # generate profit data, include the new items
        self.profit_data.update(item_profit)

        self.new_items = [i for i in self.new_items if i not in sampled_items]

        for u in set(self.ratings_df["userId"]):
            user_ratings = self.ratings_df[self.ratings_df["userId"] == u]
            b_u = self.ratings_df["rating"].mean() + (
                user_ratings["rating"].mean() - self.ratings_df["rating"].mean()
            )

            self.ratings_df = self.ratings_df._append(
                pd.DataFrame(
                    {
                        "userId": [u for i in range(len(sampled_items))],
                        "movieId": sampled_items,
                        "rating": [b_u for i in range(len(sampled_items))],
                    }
                ),
                ignore_index=True,
            )

            # append the new items to the tail of recommendation list
            self.recommendations[u].extend(
                [
                    {"iid": i, "rating": b_u, "profit": self.profit_data[i]}
                    for i in sampled_items
                ]
            )

        print("Done")

    def create_provider(self: object) -> None:
        """

        This function creates a provider agent and adds it to the scheduler to be activated before any agent.
        """
        provider = ProviderAgent(0, self)
        self.schedule.add(provider)

    def create_consumers(self: object) -> None:
        """

        A function creates consumer agents and adds them to the scheduler to be activated in a random order.
        """
        datapath = get_data_dir()
        initials_betapath = os.path.join(datapath, "trust", "beta_initials.p")
        if not os.path.isfile(initials_betapath):
            initialize_beta(self.ratings_df)
        initial_beta = pickle.load(open(initials_betapath, "rb"))
        for i in range(1, self.num_consumers + 1):
            beta_params = initial_beta[i]
            consumer = ConsumerAgent(i, self, beta_params, self.consumers_thresholds[i])
            self.schedule.add(consumer)

    def initialize_parameters(self: object, **kwargs: dict) -> None:
        """

        :param **kwargs: dict may store parameters values.

         This function initialize model variables using the parameters defined in the config file.
         Model variables hold values to be shared with all agents.
        """

        self.consumers_thresholds = {}
        recdata_path = get_rec_dir()
        self.recommendations = pickle.load(
            open(f"{recdata_path}/consumers_items_utilities_predictions.p", "rb")
        )
        self.predictive_model = pickle.load(open(f"{recdata_path}/SVDmodel.p", "rb"))

        self.ratings_df = get_ratings_data()
        self.recommendation_strategy = kwargs["recommendation_strategy"]
        self.switch_to_strategy = None

        self.quantile_consumer_expectation = kwargs["quantile_consumer_expectation"]
        self.recommendation_length = model_parameters["recommendation_length"]
        self.seed = next(self.c)

        self.profit_data = generate_profitdata(self.seed, get_items())

        self.num_consumers = self.ratings_df["userId"].nunique()
        self.num_items = get_num_items
        self.user_consumed_items = defaultdict(list)
        self.time = model_parameters["timesteps"]
        self.runs = model_parameters["number_of_runs"]
        self.feedback_likelihood = model_parameters["feedback_likelihood"]
        if self.seed == self.runs:
            setattr(RecommendationModel, "c", itertools.count(1))

        # get new items
        data_path = get_dataset_dir()
        new_items_path = os.path.join(data_path, model_input["extra_items_dataset"])
        new_items_df = pd.read_csv(new_items_path)
        self.new_items = list(new_items_df["movieId"])

        # compute the consumers" expectation thresholds
        self.compute_thresholds()
        # get the recommendations
        self.get_precomputed_consumers_utilities(0)

        # self.social_media = [
        #     [0, 0] for _ in range(model_parameters["timesteps"] + 1)
        # ]  # [number_of_likes, number_of dislikes]

        self.social_media = [0, 0]

        self.a = 1
        self.social_reputation = 0
        self.pos_posts = 0
        self.neg_posts = 0

        self.dropout_consumers = []
        self.topn = None
        # the output of these variables are taken from the provider agent
        self.total_profit = 0
        self.number_of_consumption = 0
        # self.social_reputation = None
        self.avg_profit_per_consumption = 0
        self.total_profit_history = [0]
        self.social_reputation_history = [0]

        # thresholds to switch strategies
        self.social_reputation_thresh = model_parameters["reputation_threshold"]
        self.profit_thresh = model_parameters["profit_threshold"]

    def compute_thresholds(self: object) -> None:
        """

        This function computes the expectation threshold for each consumer by taking the quantile value of the items ranked
        descendingly according to their perceived utilities to each consumer.
        """
        print("Compute consumer's expectation thresholds...")
        for c, recs in self.recommendations.items():
            ratings_c = [r["rating"] for r in recs]
            self.consumers_thresholds[c] = np.quantile(
                np.sort(ratings_c), self.quantile_consumer_expectation
            )
        print("Done")

    def update_consumer_thresholds(self: object) -> None:
        """

        This function adds precomputed the expectation thresholds for each consumer.
        """
        self.compute_thresholds()
        consumers_class = list(self.schedule.agents_by_type.keys())[1]
        for a in self.schedule.agents_by_type[consumers_class]:
            a.minimum_utility_threshold = self.consumers_thresholds[a.consumer_id]

    def get_precomputed_consumers_utilities(self, i: bool) -> None:
        """
        :param i: bool as flag to distinguish initial consumer utilities or an update of the predictions.
         A function loads the precomputed consumers items" utilities or update the previous utilities.
        """
        weights = [[0.5, 0.5], [0, 1], [0.9, 0.1]]
        recdata_path = get_rec_dir()
        if self.recommendation_strategy == "consumer-centric":
            pass

        elif self.recommendation_strategy == "balanced":
            self.recommendations = rerank_items_consider_profit(
                self.recommendations, self.profit_data, weights[0]
            )

        elif self.recommendation_strategy == "profit-centric":
            self.recommendations = rerank_items_consider_profit(
                self.recommendations, self.profit_data, weights[1]
            )

        elif self.recommendation_strategy == "consumer-biased":
            self.recommendations = rerank_items_consider_profit(
                self.recommendations, self.profit_data, weights[2]
            )

        else:
            if self.recommendation_strategy == "popular-correlated-profit":
                self.profit_data = generate_profit_data_popularity(
                    self.seed, get_items()
                )
            popular_items = get_popular_items(get_ratings_data())
            self.recommendations = get_predictions_popular_items(
                self.num_consumers, popular_items, self.predictive_model
            )

    def update_provider_utilities(self: object, item: int) -> None:
        """

        :param item: int represents the consumed item.

         This function summed up the profit gained from consumed items
        """
        # get the provider instance from the scheduler
        provider_class = list(self.schedule.agents_by_type.keys())[0]
        provider = self.schedule.agents_by_type[provider_class][0]
        provider.update_provider_utilities(item)
        self.total_profit = provider.total_profit_of_consumed_items
        self.number_of_consumption = provider.number_of_consumption
        self.avg_profit_per_consumption = provider.avg_profit_per_consumption

    def update_predictions(self: object) -> None:
        """

        A function predicts consumers' utilities periodically in the simulation.
        considering consumers' feedback
        """
        predictions, predictive_model = predict_consumers_items_utilities(
            self.ratings_df
        )
        self.predictive_model = predictive_model
        self.recommendations = get_ordered_recs(predictions)
        self.get_precomputed_consumers_utilities(1)
        self.remove_consumed_items()

    def recompute_consumers_utilities(self: object) -> None:
        """

        A function to replace the predicted consumers utilities with the true utilities of the consumed items
        """
        print("Recomputing consumers predicted utilities...")
        temp_df = pd.DataFrame()
        for k, vlist in self.user_consumed_items.items():
            for v in vlist:
                # if the flag of sending feedback is on
                if v["feedback"]:
                    row = {
                        "userId": k,
                        "movieId": v["iid"],
                        "rating": rescale_rating(v["rating"]),
                    }
                    temp_df._append(row, ignore_index=True)

        self.ratings_df = self.ratings_df._append(temp_df, ignore_index=True)
        self.update_predictions()
        self.user_consumed_items = defaultdict(list)

    def remove_consumed_items(self: object) -> None:
        """

        A function to remove the consumed items for each consumer to make sure consumers receive unique recommendations
        """
        for uid, recs in self.recommendations.items():
            s = set([(x["iid"]) for x in self.user_consumed_items[uid]])
            recs = [x for x in recs if x["iid"] not in s]
            self.recommendations[uid] = recs
        self.user_consumed_items = defaultdict(list)

    def store_consumed_items(self: object, consumer_id: int, item_id: int) -> None:
        """

        :param consumer_id: A consumer id who consumed an item
        :param item_id: An id of the selected item

         A function to store consumed items for each consumer in a dict
        """
        self.user_consumed_items[consumer_id].append(item_id)

    def compute_social_reputation(self):

        smooth = model_parameters["social_media_smooth_min_rate"] * (
            model_parameters["social_media_smooth_max_rate"]
            / model_parameters["social_media_smooth_min_rate"]
        ) ** (self.schedule.steps / model_parameters["timesteps"])

        self.social_media[0] = self.social_media[0] + smooth * (
            self.pos_posts - self.social_media[0]
        )
        self.social_media[1] = self.social_media[1] + smooth * (
            self.neg_posts - self.social_media[1]
        )

        # pos = sum([p[0] for p in self.social_media[: self.schedule.steps]])

        # neg = sum([p[1] for p in self.social_media[: self.schedule.steps]])

        rep = self.social_media[0] / (self.social_media[0] + self.social_media[1] + 1)

        if self.schedule.steps == 1:
            rep = rep * smooth

        return rep

        # decay_rate = 0.001
        # pos = np.sum(
        #     [
        #         self.social_media[t_i][0]
        #         * np.exp(-decay_rate * (self.schedule.steps - t_i))
        #         for t_i in range(self.schedule.steps + 1)
        #     ]
        # )
        # neg = np.sum(
        #     [
        #         self.social_media[t_i][1]
        #         * np.exp(-decay_rate * (self.schedule.steps - t_i))
        #         for t_i in range(self.schedule.steps + 1)
        #     ]
        # )
        # return pos / (pos + neg+1)

    def step(self):
        """

        A function to handle model adaptation each time step
        """
        if model_parameters["add_new_item"]:
            if (self.schedule.steps + 1) % model_parameters[
                "frequency_adding_items"
            ] == 0 and self.schedule.steps + 1 != model_parameters["timesteps"]:
                self.add_new_items()

        t0 = time.process_time()
        if (self.schedule.steps + 1) % model_parameters[
            "frequency_update_expectation"
        ] == 0 and self.schedule.steps + 1 != model_parameters["timesteps"]:
            self.update_consumer_thresholds()

        # recompute consumers' utilities
        # if model_parameters["update_utilities"]:
        #     if (self.schedule.steps + 1) % model_parameters[
        #         "frequency_recompute_utilities"
        #     ] == 0 and self.schedule.steps + 1 != model_parameters["timesteps"]:
        #         self.recompute_consumers_utilities()

        # recompute the utilities
        if model_parameters["update_utilities"]:
            if self.schedule.steps + 1 == (model_parameters["timesteps"] / 2):
                self.recompute_consumers_utilities()

        if self.schedule.steps > 0:
            num_posts = (
                # self.social_media[self.schedule.steps - 1][0]
                # + self.social_media[self.schedule.steps - 1][1]
                self.social_media[0]
                + self.social_media[1]
            )
            self.a = min((num_posts / (model_parameters["numposts_threshold"])), 1)

        self.schedule.step()

        self.social_reputation = self.compute_social_reputation()

        # if there is a switch instead of only take the recent posts, consider taking a percentage of the previous posts

        self.total_profit_history.append(self.total_profit)
        self.social_reputation_history.append(self.social_reputation)

        self.pos_posts = 0
        self.neg_posts = 0

        # remove dropout consumers from the platform
        for c in self.dropout_consumers:
            c.active = False
            self.schedule.remove(c)
            # self.dropout_consumers = []

        # collect data
        self.datacollector.collect(self)
        t1 = time.process_time()
        print(f"step {self.schedule.steps}, time spent: {t1 - t0}")
