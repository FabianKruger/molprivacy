"""Implementation of the RMIA attack."""

import numpy as np

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.metrics.attack_result import CombinedMetricResult
from leakpro.signals.signal import ModelLogits
from leakpro.user_inputs.abstract_input_handler import AbstractInputHandler
from leakpro.utils.extract_true_positives import (
    write_true_positives_to_disc,
    compare_datasets,
)
from torch.utils.data import Subset
from sklearn.metrics import roc_curve


class AttackRMIA(AbstractMIA):
    """Implementation of the RMIA attack."""

    def __init__(self, handler: AbstractInputHandler, configs: dict) -> None:
        """Initialize the RMIA attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)
        self.shadow_models = []
        self.signal = ModelLogits()
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None
        self.configs = configs
        self.signal_values = None
        self.true_labels = None

        self.logger.info("Configuring RMIA attack")
        self._configure_attack(configs)

    def _configure_attack(self, configs: dict) -> None:
        """Configure the RMIA attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.num_shadow_models = configs.get("num_shadow_models", 4)
        self.offline_a = configs.get("data_fraction", 0.33)
        self.offline_b = configs.get("offline_b", 0.66)
        self.gamma = configs.get("gamma", 2.0)
        self.temperature = configs.get("temperature", 2.0)
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.online = configs.get("online", False)
        self.attack_data_fraction = configs.get("attack_data_fraction", 0.1)

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "offline_a": (self.offline_a, 0, 1),
            "offline_b": (self.offline_b, 0, 1),
            "gamma": (self.gamma, 0, None),
            "temperature": (self.temperature, 0, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "attack_data_fraction": (self.attack_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "The RMIA attack is a membership inference attack based on the output logits of a black-box model."
        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to compute the likelihood LR_z of p(z|theta) to p(z) for the target model.\
            2. The ratio is used to compute the likelihood ratio LR_x of p(x|theta) to p(x) for the target model. \
            3. The ratio LL_x/LL_z is viewed as a random variable (z is random) and used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def softmax(
        self,
        all_logits: np.ndarray,
        true_label_indices: np.ndarray,
        return_full_distribution: bool = False,
    ) -> np.ndarray:
        """Compute the softmax function.

        Args:
        ----
            all_logits (np.ndarray): Logits for each class.
            true_label_indices (np.ndarray): Indices of the true labels.
            return_full_distribution (bool, optional): return the full distribution or just the true class probabilities.

        Returns:
        -------
            np.ndarray: Softmax output.

        """
        logit_signals = all_logits / self.temperature
        max_logit_signals = np.max(logit_signals, axis=2)
        logit_signals = logit_signals - max_logit_signals.reshape(1, -1, 1)
        exp_logit_signals = np.exp(logit_signals)
        exp_logit_sum = np.sum(exp_logit_signals, axis=2)

        if return_full_distribution is False:
            true_exp_logit = exp_logit_signals[
                :, np.arange(exp_logit_signals.shape[1]), true_label_indices
            ]
            output_signal = true_exp_logit / exp_logit_sum
        else:
            output_signal = exp_logit_signals / exp_logit_sum[:, :, np.newaxis]
        return output_signal

    def prob_binary(
        self, all_logits: np.ndarray, true_label_indices: np.ndarray
    ) -> np.ndarray:
        """Compute the sigmoid function for binary classification with one output node."""
        # apply sigmoid function
        prob_class_1 = self.sigmoid(all_logits)
        p_x_given_model = np.where(
            true_label_indices == 1, prob_class_1, 1 - prob_class_1
        )
        return p_x_given_model

    def sigmoid(self, x):
        # Create a mask for positive and negative values
        positive_mask = x >= 0
        negative_mask = ~positive_mask

        # Initialize the result array with the same shape as x
        result = np.zeros_like(x, dtype=np.float64)

        # Apply the numerically stable sigmoid for positive values
        result[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))

        # Apply the numerically stable sigmoid for negative values
        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x / (1.0 + exp_x)

        return result

    def prepare_attack(self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """
        self.logger.info("Preparing shadow models for RMIA attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        self.logger.info("Preparing attack data for training the RMIA attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(
            include_train_indices=self.online, include_test_indices=self.online
        )

        # train shadow models
        self.logger.info(
            f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)"
        )
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models=self.num_shadow_models,
            shadow_population=self.attack_data_indices,
            training_fraction=self.training_data_fraction,
            online=self.online,
        )
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(
            self.shadow_model_indices
        )

        # compute quantities that are not touching the audit dataset
        if self.online is False:
            # compute the ratio of p(z|theta) (target model) to p(z)=sum_{theta'} p(z|theta') (shadow models)
            # for all points in the attack dataset output from signal: # models x # data points x # classes

            # subsample the attack data based on the fraction
            self.logger.info(
                f"Subsampling attack data from {len(self.attack_data_indices)} points"
            )
            n_points = int(self.attack_data_fraction * len(self.attack_data_indices))
            attack_data = self.sample_data_from_dataset(
                self.attack_data_indices, n_points
            ).dataset
            self.logger.info(
                f"Number of attack data points after subsampling: {len(attack_data)}"
            )

            # get the true label indices
            z_true_labels = np.array(attack_data._labels)

            # run points through real model to collect the logits
            logits_theta = np.array(
                self.signal([self.target_model], attack_data, handler=self.handler)
            )
            # collect the softmax output of the correct class
            p_z_given_theta = self.softmax(logits_theta, z_true_labels)

            # run points through shadow models and collect the logits
            logits_shadow_models = self.signal(
                self.shadow_models, attack_data, handler=self.handler
            )
            # collect the softmax output of the correct class for each shadow model
            p_z_given_shadow_models = [
                self.softmax(np.array(x).reshape(1, *x.shape), z_true_labels)
                for x in logits_shadow_models
            ]
            # stack the softmax output of the correct class for each shadow model to dimension # models x # data points
            p_z_given_shadow_models = np.array(p_z_given_shadow_models).squeeze()

            # evaluate the marginal p(z)
            p_z = (
                np.mean(p_z_given_shadow_models, axis=0)
                if len(self.shadow_models) > 1
                else p_z_given_shadow_models.squeeze()
            )
            p_z = 0.5 * ((self.offline_a + 1) * p_z + (1 - self.offline_a))

            self.ratio_z = p_z_given_theta / (p_z + self.epsilon)

    def _online_attack(self) -> None:
        self.logger.info("Running RMIA online attack")

        # get the logits for the audit dataset
        # give a numpy array of all the correct labels instead
        # TODO: ask Johan if this works with shuffle = True
        audit_data = self.handler.get_dataset(self.audit_dataset["data"])
        compare_datasets(audit_data, self.configs)
        labels_list = []
        for _, labels in audit_data:
            labels_list.append(labels.item())
        # Convert the list to a numpy array
        ground_truth_indices = np.array(labels_list)
        # ground_truth_indices = np.array(audit_data._labels)

        # find the shadow models that are trained on what points in the audit dataset
        in_indices_mask = (
            ShadowModelHandler()
            .get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])
            .T
        )

        # filter out the points that no shadow model has seen and points that all shadow models have seen
        num_shadow_models_seen_points = np.sum(in_indices_mask, axis=0)
        # make sure that the audit points are included in the shadow model training (but not all)
        mask = (num_shadow_models_seen_points > 0) & (
            num_shadow_models_seen_points < self.num_shadow_models
        )
        audit_data = Subset(audit_data, (np.where(mask)[0]))
        if len(audit_data) == 0:
            raise ValueError(
                "No points in the audit dataset are used for the shadow models"
            )

        # find out how many in-members survived the filtering
        in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
        # find out how many out-members survived the filtering
        num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
        out_members = np.arange(len(in_members), len(in_members) + num_out_members)
        out_model_indices = ~in_indices_mask[:, mask]

        self.logger.info(
            f"Number of points in the audit dataset that are used for online attack: {len(audit_data)}"
        )

        # run points through target model to get logits
        logits_theta = np.squeeze(
            np.array(self.signal([self.target_model], audit_data, handler=self.handler))
        )
        # collect the softmax output of the correct class
        # p_x_given_target_model = P(Y=correct_class)
        p_x_given_target_model = self.prob_binary(
            logits_theta, ground_truth_indices[mask]
        )
        # p_x_given_target_model = self.softmax(logits_theta, ground_truth_indices)

        # run points through shadow models, colelct logits and compute p(x)
        logits_shadow_models = self.signal(
            self.shadow_models, audit_data, handler=self.handler
        )
        p_x_given_shadow_models = np.array(
            [
                self.prob_binary(x.squeeze(), ground_truth_indices[mask])
                for x in logits_shadow_models
            ]
        )

        # p_x_given_shadow_models = [self.softmax(np.array(x).reshape(1,*x.shape), ground_truth_indices)
        #                             for x in logits_shadow_models]
        # p_x_given_shadow_models = np.array(p_x_given_shadow_models).squeeze()
        p_x = (
            np.mean(p_x_given_shadow_models, axis=0)
            if len(self.shadow_models) > 1
            else p_x_given_shadow_models.squeeze()
        )
        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        # Make a "random sample" to compute p(z) for points in attack dataset on the OUT shadow models for each audit point
        self.attack_data_index = self.sample_indices_from_population(
            include_train_indices=False, include_test_indices=False
        )

        # subsample the attack data based on the fraction
        self.logger.info(
            f"Subsampling attack data from {len(self.attack_data_index)} points"
        )
        self.attack_data_index = np.random.choice(
            self.attack_data_index,
            int(self.attack_data_fraction * len(self.attack_data_index)),
            replace=False,
        )
        self.logger.info(
            f"Number of attack data points after subsampling: {len(self.attack_data_index)}"
        )
        attack_data = self.handler.get_dataset(self.attack_data_index)
        # attack_data = self.population.subset(self.attack_data_index)
        # get the true label indices
        labels_list = []
        for _, labels in attack_data:
            labels_list.append(labels.item())
        # Convert the list to a numpy array
        z_true_labels = np.array(labels_list)
        # run points through real model to collect the logits
        logits_target_model = np.array(
            self.signal([self.target_model], attack_data, handler=self.handler)
        ).squeeze()
        # collect the softmax output of the correct class
        p_z_given_target_model = self.prob_binary(logits_target_model, z_true_labels)
        # p_z_given_target_model = self.softmax(logits_target_model, z_true_labels)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(
            self.shadow_models, attack_data, handler=self.handler
        )
        # collect the softmax output of the correct class for each shadow model
        # TODO: adapt
        p_z_given_shadow_models = np.array(
            [self.prob_binary(x.squeeze(), z_true_labels) for x in logits_shadow_models]
        )

        # evaluate the marginal p(z) by averaging over the OUT models
        p_z = np.zeros((len(audit_data), len(attack_data)))
        # change low values to something suitable with float16 precision
        for i in range(len(audit_data)):
            model_mask = out_model_indices[:, i]
            p_z[i] = np.mean(p_z_given_shadow_models[model_mask, :], axis=0)
        ratio_z = p_z_given_target_model / (p_z + self.epsilon)

        # for each x, compute the score
        likelihoods = ratio_x.reshape(-1, 1) / ratio_z
        score = np.mean(likelihoods > self.gamma, axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1, 1)
        self.out_member_signals = score[out_members].reshape(-1, 1)

        signal_values = np.concatenate(
            [self.in_member_signals, self.out_member_signals]
        )

        true_labels = np.concatenate(
            [
                np.ones(len(self.in_member_signals)),
                np.zeros(len(self.out_member_signals)),
            ]
        )
        self.signal_values = signal_values
        self.true_labels = true_labels
        write_true_positives_to_disc(
            dataset=audit_data,
            scores=signal_values,
            labels=true_labels,
            mask=mask,
            configs=self.configs,
            attack="rmia",
        )

    def _offline_attack(self) -> None:
        self.logger.info("Running RMIA offline attack")
        # get the logits for the audit dataset
        audit_data = self.get_dataloader(self.audit_dataset["data"]).dataset
        ground_truth_indices = np.array(audit_data._labels)

        # run target points through real model to get logits
        logits_theta = np.array(
            self.signal([self.target_model], audit_data, handler=self.handler)
        )
        # collect the softmax output of the correct class
        p_x_given_target_model = self.softmax(logits_theta, ground_truth_indices)

        # run points through shadow models and collect the logits
        logits_shadow_models = self.signal(
            self.shadow_models, audit_data, handler=self.handler
        )
        # collect the softmax output of the correct class for each shadow model
        p_x_given_shadow_models = [
            self.softmax(np.array(x).reshape(1, *x.shape), ground_truth_indices)
            for x in logits_shadow_models
        ]
        # stack the softmax output of the correct class for each shadow model
        # to dimension # models x # data points
        p_x_given_shadow_models = np.array(p_x_given_shadow_models).squeeze()
        # evaluate the marginal p_out(x) by averaging the output of the shadow models
        p_x_out = (
            np.mean(p_x_given_shadow_models, axis=0)
            if len(self.shadow_models) > 1
            else p_x_given_shadow_models.squeeze()
        )

        # compute the marginal p(x) from P_out and p_in where p_in = a*p_out+b
        p_x = 0.5 * ((self.offline_a + 1) * p_x_out + (1 - self.offline_a))

        # compute the ratio of p(x|theta) to p(x)
        ratio_x = p_x_given_target_model / (p_x + self.epsilon)

        # for each x, compare it with the ratio of all z points
        likelihoods = ratio_x.T / self.ratio_z

        in_members = self.audit_dataset["in_members"]
        out_members = self.audit_dataset["out_members"]

        score = np.mean(likelihoods > self.gamma, axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1, 1)
        self.out_member_signals = score[out_members].reshape(-1, 1)

    def run_attack(self) -> CombinedMetricResult:
        # TODO: return dictionary instead with dataframe of tpr and fpr (use sklearn for it)
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        # perform the attack
        if self.online is True:
            self._online_attack()
        else:
            self._offline_attack()

        # We use sklearn roc curve instead here
        # create thresholds
        # min_signal_val = np.min(np.concatenate([self.in_member_signals, self.out_member_signals]))
        # max_signal_val = np.max(np.concatenate([self.in_member_signals, self.out_member_signals]))
        # thresholds = np.linspace(min_signal_val, max_signal_val, 1000)

        # member_preds = np.greater(self.in_member_signals, thresholds).T
        # non_member_preds = np.greater(self.out_member_signals, thresholds).T

        # # what does the attack predict on test and train dataset
        # predictions = np.concatenate([member_preds, non_member_preds], axis=1)
        # # set true labels for being in the training dataset
        # true_labels = np.concatenate(
        #     [
        #         np.ones(len(self.in_member_signals)),
        #         np.zeros(len(self.out_member_signals)),
        #     ]
        # )
        # signal_values = np.concatenate(
        #     [self.in_member_signals, self.out_member_signals]
        # )

        fpr, tpr, _ = roc_curve(y_true=self.true_labels, y_score=self.signal_values)
        return {"fpr": fpr, "tpr": tpr}
