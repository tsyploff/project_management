import numpy as np
from typing import Optional


def __find_best_alternative(scores, is_loss_matrix: bool) -> int:
    """
    If alternatives matrix values is losses then score should be
    minimized and vice versa.

    :param scores: criterion scores for each alternative
    :param is_loss_matrix: boolean flag loss or profit values are in matrix
    :return: number of the most preferable alternative
    """
    if is_loss_matrix:
        return np.argmin(scores)
    else:
        return np.argmax(scores)


def laplace_criterion(alternatives, is_loss_matrix: bool = True) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. The function gives the most preferable alternative with Laplace'
    criterion: min of mean values through scenarios when is_loss_matrix is True
    and max of mean values through scenarios otherwise.

    :param alternatives: N x M matrix of numeric values
    :param is_loss_matrix: boolean flag loss or profit values are in matrix
    :return: number of the most preferable alternative
    """
    return __find_best_alternative(np.mean(alternatives, axis=1), is_loss_matrix)


def wald_criterion(alternatives, is_loss_matrix: bool = True) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. The function gives the most preferable alternative with Wald'
    criterion: minimax when is_loss_matrix is True and max of min otherwise.

    :param alternatives: N x M matrix of numeric values
    :param is_loss_matrix: boolean flag loss or profit values are in matrix
    :return: number of the most preferable alternative
    """
    if is_loss_matrix:
        return np.max(alternatives, axis=1).argmin()
    else:
        return np.min(alternatives, axis=1).argmax()


def savage_criterion(alternatives, is_loss_matrix: bool = True) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. The function gives the most preferable alternative with Savage'
    criterion: Wald' criterion on regrets matrix, where regrets matrix is
    differences between best alternative in each scenario.

    :param alternatives: N x M matrix of numeric values
    :param is_loss_matrix: boolean flag loss or profit values are in matrix
    :return: number of the most preferable alternative
    """
    if is_loss_matrix:
        regrets_matrix = alternatives - np.min(alternatives, axis=0).reshape(1, -1)
        return wald_criterion(regrets_matrix)
    else:
        regrets_matrix = np.max(alternatives, axis=0).reshape(1, -1) - alternatives
        return wald_criterion(regrets_matrix)


def hurwitz_criterion(alternatives, is_loss_matrix: bool = True, alpha: float = .5) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. The function gives the most preferable alternative with Hurwitz'
    criterion. It's like Laplace but instead of mean convex combination of min
    and max used.

    :param alternatives: N x M matrix of numeric values
    :param is_loss_matrix: boolean flag loss or profit values are in matrix
    :param alpha: coefficient in convex combination
    :return: number of the most preferable alternative
    """
    upper_values = np.max(alternatives, axis=1)
    lower_values = np.min(alternatives, axis=1)
    scores = (1 - alpha) * upper_values + alpha * lower_values
    return __find_best_alternative(scores, is_loss_matrix)


def __frequencies_to_probabilities(frequencies, default: float = .0):
    """
    Normalizes frequencies in such way that sum(frequencies) = 1 and
    fill Nones with default.

    :param frequencies: list of numeric values
    :return: normalized probabilities
    """
    probabilities = np.nan_to_num(frequencies, nan=default)

    if np.isnan(probabilities).any() or np.any(probabilities < 0) or np.sum(probabilities) <= 0:
        raise ValueError("Frequencies must all be non negative numeric values and sum must be positive." +
                         f"Got {frequencies}.")

    return probabilities / np.sum(probabilities)


def expectation_criterion(alternatives, frequencies, is_loss_matrix: bool = True) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. Let frequencies[j] be the frequency of j scenario.

    The function gives the most preferable alternative with expectation
    criterion: min of expectation values through scenarios when is_loss_matrix
    is True and max of expectation values through scenarios otherwise.

    :param alternatives: N x M matrix of numeric values
    :param frequencies: probabilities of each scenario
    :param is_loss_matrix: boolean flag loss or profit values are in matrix
    :return: number of the most preferable alternative
    """
    probabilities = __frequencies_to_probabilities(frequencies)
    return __find_best_alternative(np.dot(alternatives, probabilities), is_loss_matrix)


def std_criterion(alternatives, frequencies) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. Let frequencies[j] be the frequency of j scenario.

    The function gives the most preferable alternative with standard deviation
    criterion: min of standard deviation regardless of the parameter is_loss_matrix

    :param alternatives: N x M matrix of numeric values
    :param frequencies: probabilities of each scenario
    :return: number of the most preferable alternative
    """
    probabilities = __frequencies_to_probabilities(frequencies)
    scores = np.dot(alternatives ** 2, probabilities) - np.dot(alternatives, probabilities) ** 2
    return np.argmin(scores)


def variation_coefficient_criterion(alternatives, frequencies) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. Let frequencies[j] be the frequency of j scenario.

    The function gives the most preferable alternative with coefficient of variation
    criterion: min of coefficient of variation regardless of the parameter is_loss_matrix

    :param alternatives: N x M matrix of numeric values
    :param frequencies: probabilities of each scenario
    :return: number of the most preferable alternative
    """
    probabilities = __frequencies_to_probabilities(frequencies)
    expectation = np.dot(alternatives, probabilities)
    var = np.dot(alternatives ** 2, probabilities) - expectation ** 2
    scores = np.sqrt(var) / np.abs(expectation)
    return np.argmin(scores)


def linear_combination_mean_std_criterion(
        alternatives,
        frequencies,
        is_loss_matrix: bool = True,
        alpha: float = .5,
        beta: Optional[float] = None
) -> int:
    """
    Let alternatives be the NxM matrix with N alternatives and M scenarios.
    Let alternatives[i, j] be the loss or profit value of alternative i in
    scenario j. Let frequencies[j] be the frequency of j scenario.

    The function gives the most preferable alternative with criterion of
    linear combination of mean and std:

    alpha * mean + beta * std -> min if is_loss_matrix is True
    alpha * mean - beta * std -> max otherwise

    :param alternatives: N x M matrix of numeric values
    :param frequencies: probabilities of each scenario
    :param is_loss_matrix: fictitious boolean flag loss or profit values are in matrix
    :param alpha: coefficient at mean
    :param beta: coefficient at std, = 1 - alpha by default
    :return: number of the most preferable alternative
    """
    probabilities = __frequencies_to_probabilities(frequencies)
    expectation = np.dot(alternatives, probabilities)
    var = np.dot(alternatives ** 2, probabilities) - expectation ** 2
    std = np.sqrt(var)

    b = (1 - alpha) if beta is None else beta

    if is_loss_matrix:
        return np.argmin(alpha * expectation + b * std)
    else:
        return np.argmax(alpha * expectation - b * std)
