from mesa import Agent
from mesa.model import Model
from read_config import *
from utils import *
import random
import numpy as np


class ProviderAgent(Agent):
    def __init__(self: Agent, id: int, model: Model) -> None:
        super().__init__(id, model)
        self.recommendation_strategy = None
        self.recommendations = model.recommendations
        self.total_profit_of_consumed_items = 0
        self.number_of_consumption = 0
        self.avg_profit_per_consumption = 0

    def update_provider_utilities(self: Agent, item: int) -> None:
        """

        :param item: int of item id.

        A function computes service provider"s utilities.
        """
        self.total_profit_of_consumed_items += self.model.profit_data[item["iid"]]
        self.number_of_consumption += 1
        self.avg_profit_per_consumption = (
            self.total_profit_of_consumed_items / self.number_of_consumption
        )

    # def get_total_profit(self) -> None:
    #     return self.total_profit

    def reset_provider_utilities(self) -> None:
        """

        This function resets the utilities of the service provider.
        """
        self.total_profit_of_consumed_items = 0
        self.number_of_consumption = 0
        self.avg_profit_per_consumption = 0

    def apply_recommendation_strategy(self) -> None:
        """

        A function sets a recommendation recommendation_strategy to be applied on the recommended items.
         Five strategies were implemented and one is only used in each simulation run:
         consumer_only: Recommened items to maximize consumer items' utilities.
         balance_equal_weights: Recommened items while balancing consumer items" utilities and provider's utility (profit) using equal weights.
         profit_only: Recommend items to maximize provider's utility.
         balance_unequal_weights: Recommened items while balancing consumer items' utilities and provider's utility (profit) using a higher weight on consumers items' utilities.
         popular_based: Recommened items with the highest number of ratings.
        """
        self.recommendation_strategy = self.model.recommendation_strategy
        self.model.topn = get_top_n(
            self.model.recommendations, model_parameters["recommendation_length"]
        )

    def get_previous_data_avg(self, data):
        # moving averages to compute the threshold
        w = model_parameters["n"]
        t = self.model.schedule.steps
        if t < w and data:
            return data[t]
        w_data = data[t - w : t + 1]
        ma = np.mean(w_data)
        return np.round(ma, 3)

    def switch_strategy(self):
        """
        Switch strategies based on given thresholds of socisl reputation and profit
        """
        social_reputation_previos_data_avg = self.get_previous_data_avg(
            self.model.social_reputation_history
        )
        profit_previos_data_avg = self.get_previous_data_avg(
            self.model.total_profit_history
        )

        if (self.model.schedule.steps + 1) % model_parameters["period_switch"] == 0:
            if social_reputation_previos_data_avg < self.model.social_reputation_thresh:
                self.model.switch_to_strategy = "consumer-centric"

                self.model.recommendations = get_ordered_recs(
                    self.model.recommendations
                )

            elif profit_previos_data_avg < self.model.profit_thresh:
                self.model.switch_to_strategy = "profit-centric"

                rerank_items_consider_profit(
                    self.model.recommendations, self.model.profit_data, [0, 1]
                )

    def step(self):
        self.reset_provider_utilities()
        self.apply_recommendation_strategy()
        if model_parameters["switch_strategy"]:
            self.switch_strategy()
