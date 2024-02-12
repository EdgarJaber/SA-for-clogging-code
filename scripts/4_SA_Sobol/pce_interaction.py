from collections import defaultdict
import openturns as ot


def ComputeGroupLabelsFromLabelNames(variableNamesList, groupsList):
    """
    Compute the labels of groups from names

    Parameters
    ----------
    variableNamesList : list(str)
        The list of variables names.

    groupsList: list(list(int))
        Each item in the list is a list of indices of variables
        representing a group.
        The number of items in the list is the number of
        groups.

    Returns
    -------
    group_labels : list(str)
        The labels of the groups of variables.

    Example
    -------
    >>> variableNamesList = ["A", "B", "C"]
    >>> groupsList = [[0], [1], [2], [0, 1], [0, 2]]
    >>> group_labels = ComputeGroupLabelsFromLabelNames(variableNamesList, groupsList)
    >>> group_labels
    ["A", "B", "C", "AB", "AC"]
    """
    number_of_keys = len(variableNamesList)
    group_labels = []
    for i in range(len(groupsList)):
        group = groupsList[i]
        for j in group:
            if j < 0 or j >= number_of_keys:
                raise ValueError(
                    f"index = {j} inconsistent with the number of keys {number_of_keys}"
                )
        label_list = [variableNamesList[j] for j in group]
        label = "[" + ",".join(label_list) + "]"
        group_labels.append(label)
    return group_labels
    

def ComputeInteractionSobolIndices(polynomialChaosResult):
    """
    Compute Sobol' interaction indices of a PCE

    Parameters
    ----------
    polynomialChaosResult : ot.FunctionalChaosResult
        The polynomial chaos expansion.

    Returns
    -------
    interaction_sobol_indices : ot.Sample(number_of_groups, outputDimension)
        The interaction Sobol' indices.

    group_list : list(list(int))
        The list of groups of variables.
        Each group of variables is a list of indices representing
        the index of each variable in the group.
    """
    outputSample = polynomialChaosResult.getOutputSample()
    outputDimension = outputSample.getDimension()

    # Get the part of variance of each multi-index
    functionalChaosSobolIndices = ot.FunctionalChaosSobolIndices(polynomialChaosResult)
    input_dimension = polynomialChaosResult.getInputSample().getDimension()
    indices = polynomialChaosResult.getIndices()
    number_of_coefficients = len(indices)

    # Compute the part of variance of each coefficient
    part_of_variances_by_coefficient = ot.Sample(number_of_coefficients, outputDimension)
    for marginalIndex in range(outputDimension):
        part_of_variance_point = functionalChaosSobolIndices.getPartOfVariance(marginalIndex)
        part_of_variances_by_coefficient[:, marginalIndex] = ot.Sample.BuildFromPoint(part_of_variance_point)

    # Gather each multi-indice into groups of variables
    # Each key of the dictionnary represent the group of variables.
    # Each corresponding value is the list of multi-indices in that group.
    orthogonalBasis = polynomialChaosResult.getOrthogonalBasis()
    enumerateFunction = orthogonalBasis.getEnumerateFunction()
    group_dictionnary = defaultdict(list)
    for i in range(number_of_coefficients):
        multi_indices = enumerateFunction(indices[i])
        if sum(multi_indices) > 0:
            # Compute the group of input active variable of each multi-index
            group_of_variables = [j for j in range(input_dimension) if multi_indices[j] > 0]
            group_of_variables = tuple(group_of_variables)
            group_dictionnary[group_of_variables].append(i)

    """
    group_dictionnary
    defaultdict(list,
                {(0,): [1, 7],
                (1,): [2],
                (3,): [3, 8, 12, 28],
                (4,): [4, 9, 13, 29],
                (5,): [5, 10, 14],
                (6,): [6, 11, 15, 30],
                (0, 4): [16],
                (0, 5): [17],
                (0, 6): [18],
                (1, 4): [19],
                (1, 6): [20],
                (2, 5): [21],
                (3, 4): [22],
                (3, 5): [23],
                (3, 6): [24],
                (4, 5): [25],
                (4, 6): [26],
                (5, 6): [27]})
    """

    # Get the list of (unique) groups of variables
    group_list = sorted(group_dictionnary)
    number_of_groups = len(group_list)

    # Gather the interaction Sobol' indices time series
    interaction_sobol_indices = ot.Sample(number_of_groups, outputDimension)
    for i in range(number_of_groups):
        group = group_list[i]
        multi_indices_list = group_dictionnary[group]
        for coefficient_index in multi_indices_list:
            interaction_sobol_indices[i,:] += part_of_variances_by_coefficient[coefficient_index,:]
    return interaction_sobol_indices, group_list


