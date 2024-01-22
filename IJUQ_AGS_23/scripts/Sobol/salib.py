# -*- coding: utf-8 -*-
"""
A collection of functions to estimate Sobol' indices.

References
----------
Dumas A., Lois asymptotiques des estimateurs des indices de Sobol’, 
Technical report, Phimeca, 2018. pdf

"""
import openturns as ot
import numpy as np
import math
import itertools
from collections import defaultdict
import treemap as tm


def sampleDotProduct(sampleA, sampleB):
    """
    Computes the dot product of two 1D-samples.

    Parameters
    ----------
    sampleA : ot.Sample(size, 1)
        A sample.
    sampleB : ot.Sample(size, 1)
        A sample.

    Returns
    -------
    y : float
        The dot product of sampleA and sampleB.

    """
    sampleA_point = sampleA.asPoint()
    sampleB_point = sampleB.asPoint()
    y = sampleA_point.dot(sampleB_point)
    return y


def martinezSobolIndices(inputSampleA, inputSampleB, gFunction):
    """
    Given two independent input samples, compute Sobol' indices.

    Parameters
    ----------
    inputSampleA : ot.Sample(size, dimension)
        An input sample.
    inputSampleB : ot.Sample(size, dimension)
        An input sample, independent from inputSampleA.
    gFunction : ot.Function
        A function.

    Returns
    -------
    firstOrderIndices : ot.Point(dimension)
        The first order Sobol' indices.
    totalIndices : ot.Point(dimension)
        The total order Sobol' indices.

    """

    def MartinezKernel(outputSampleB, outputSampleE):
        return ot.CorrelationAnalysis(
            outputSampleB, outputSampleE
        ).computePearsonCorrelation()[0]

    dim = inputSampleA.getDimension()
    outputSampleA = gFunction(inputSampleA)
    outputSampleB = gFunction(inputSampleB)
    firstOrderIndices = ot.Point(dim)
    totalIndices = ot.Point(dim)
    for i in range(dim):
        inputSampleE = ot.Sample(inputSampleA)
        inputSampleE[:, i] = inputSampleB[:, i]
        outputSampleE = gFunction(inputSampleE)
        firstOrderIndices[i] = MartinezKernel(outputSampleB, outputSampleE)
        totalIndices[i] = 1.0 - MartinezKernel(outputSampleA, outputSampleE)
    return firstOrderIndices, totalIndices


def martinezSobolGroupIndices(inputSampleA, inputSampleB, gFunction, group):
    """
    Given two independent input samples, compute Sobol' indices of a group.

    Parameters
    ----------
    inputSampleA : ot.Sample(size, dimension)
        An input sample.
    inputSampleB : ot.Sample(size, dimension)
        An input sample, independent from inputSampleA.
    gFunction : ot.Function
        A function.
    group : list(int)
        The list of marginal indices of the inputs in the group.

    Returns
    -------
    firstOrderIndices : float
        The first order closed Sobol' indices.
    totalIndices : float
        The total order Sobol' indices.

    """
    outputSampleA = gFunction(inputSampleA)
    #
    dim = inputSampleA.getDimension()
    outputSampleB = gFunction(inputSampleB)
    firstOrderIndices = ot.Point(dim)
    totalIndices = ot.Point(dim)
    inputSampleE = ot.Sample(inputSampleA)
    for j in group:
        inputSampleE[:, j] = inputSampleB[:, j]
    outputSampleE = gFunction(inputSampleE)
    firstOrderIndices = ot.CorrelationAnalysis(
        outputSampleB, outputSampleE
    ).computePearsonCorrelation()[0]
    totalIndices = (
        1.0
        - ot.CorrelationAnalysis(
            outputSampleA, outputSampleE
        ).computePearsonCorrelation()[0]
    )
    return firstOrderIndices, totalIndices


def saltelliSobolIndices(inputSampleA, inputSampleB, gFunction):
    """
    Given two independent input samples, compute Sobol' indices.

    Parameters
    ----------
    inputSampleA : ot.Sample
        An input sample.
    inputSampleB : ot.Sample
        An input sample, independent from inputSampleA.
    gFunction : ot.Function
        A function.

    Returns
    -------
    firstOrderIndices : ot.Point(dimension)
        The first order Sobol' indices.
    totalIndices : ot.Point(dimension)
        The total order Sobol' indices.

    Reference
    ---------
    https://github.com/openturns/openturns/blob/8584beb36c7f3282147b10e83acb327da234c3e5/lib/src/Uncertainty/Algorithm/Sensitivity/SaltelliSensitivityAlgorithm.cxx
    """
    size = inputSampleA.getSize()
    dim = inputSampleA.getDimension()
    # A
    outputSampleA = gFunction(inputSampleA)
    sampleMean = outputSampleA.computeMean()[0]
    sampleVariance = outputSampleA.computeVariance()[0]
    outputSampleA -= sampleMean
    # B
    outputSampleB = gFunction(inputSampleB) - sampleMean
    firstOrderIndices = ot.Point(dim)
    totalIndices = ot.Point(dim)
    for i in range(dim):
        inputSampleE = ot.Sample(inputSampleA)
        inputSampleE[:, i] = inputSampleB[:, i]
        outputSampleE = gFunction(inputSampleE) - sampleMean
        firstOrderIndices[i] = (
            sampleDotProduct(outputSampleB, outputSampleE) / (size - 1) / sampleVariance
        )
        totalIndices[i] = (
            1.0
            - sampleDotProduct(outputSampleA, outputSampleE)
            / (size - 1)
            / sampleVariance
        )
    return firstOrderIndices, totalIndices


def jansenSobolIndices(inputSampleA, inputSampleB, gFunction):
    """
    Given two independent input samples, compute Sobol' indices.

    Parameters
    ----------
    inputSampleA : ot.Sample
        An input sample.
    inputSampleB : ot.Sample
        An input sample, independent from inputSampleA.
    gFunction : ot.Function
        A function.

    Returns
    -------
    firstOrderIndices : ot.Point(dimension)
        The first order Sobol' indices.
    totalIndices : ot.Point(dimension)
        The total order Sobol' indices.

    Reference
    ---------
    https://github.com/openturns/openturns/blob/8584beb36c7f3282147b10e83acb327da234c3e5/lib/src/Uncertainty/Algorithm/Sensitivity/JansenSensitivityAlgorithm.cxx
    """

    def JansenKernel(outputSampleB, outputSampleE, sampleVariance, size):
        yEMinusyB = outputSampleE - outputSampleB
        squaredSumyBMinusyE = sampleDotProduct(yEMinusyB, yEMinusyB)
        return 1.0 - squaredSumyBMinusyE / (2 * size - 1) / sampleVariance

    size = inputSampleA.getSize()
    dim = inputSampleA.getDimension()
    # A
    outputSampleA = gFunction(inputSampleA)
    sampleMean = outputSampleA.computeMean()[0]
    sampleVariance = outputSampleA.computeVariance()[0]
    outputSampleA -= sampleMean
    # B
    outputSampleB = gFunction(inputSampleB) - sampleMean
    firstOrderIndices = ot.Point(dim)
    totalIndices = ot.Point(dim)
    for i in range(dim):
        inputSampleE = ot.Sample(inputSampleA)
        inputSampleE[:, i] = inputSampleB[:, i]
        outputSampleE = gFunction(inputSampleE) - sampleMean
        firstOrderIndices[i] = JansenKernel(
            outputSampleB, outputSampleE, sampleVariance, size
        )
        totalIndices[i] = 1.0 - JansenKernel(
            outputSampleA, outputSampleE, sampleVariance, size
        )
    return firstOrderIndices, totalIndices


def mauntzKucherenkonSobolIndices(inputSampleA, inputSampleB, gFunction):
    """
    Given two independent input samples, compute Sobol' indices.

    Parameters
    ----------
    inputSampleA : ot.Sample
        An input sample.
    inputSampleB : ot.Sample
        An input sample, independent from inputSampleA.
    gFunction : ot.Function
        A function.

    Returns
    -------
    firstOrderIndices : ot.Point(dimension)
        The first order Sobol' indices.
    totalIndices : ot.Point(dimension)
        The total order Sobol' indices.

    Reference
    ---------
    https://github.com/openturns/openturns/blob/8584beb36c7f3282147b10e83acb327da234c3e5/lib/src/Uncertainty/Algorithm/Sensitivity/JansenSensitivityAlgorithm.cxx
    """

    size = inputSampleA.getSize()
    dim = inputSampleA.getDimension()
    # A
    outputSampleA = gFunction(inputSampleA)
    sampleMean = outputSampleA.computeMean()[0]
    sampleVariance = outputSampleA.computeVariance()[0]
    outputSampleA -= sampleMean
    # B
    outputSampleB = gFunction(inputSampleB) - sampleMean
    firstOrderIndices = ot.Point(dim)
    totalIndices = ot.Point(dim)
    for i in range(dim):
        inputSampleE = ot.Sample(inputSampleA)
        inputSampleE[:, i] = inputSampleB[:, i]
        outputSampleE = gFunction(inputSampleE) - sampleMean
        yEMinusyA = outputSampleE - outputSampleA
        firstOrderIndices[i] = (
            sampleDotProduct(yEMinusyA, outputSampleB) / (size - 1.0) / sampleVariance
        )
        totalIndices[i] = (
            -sampleDotProduct(yEMinusyA, outputSampleA) / (size - 1.0) / sampleVariance
        )
    return firstOrderIndices, totalIndices


def janonSobolIndices(inputSampleA, inputSampleB, gFunction):
    """
    Given two independent input samples, compute Sobol' indices.

    Parameters
    ----------
    inputSampleA : ot.Sample
        An input sample.
    inputSampleB : ot.Sample
        An input sample, independent from inputSampleA.
    gFunction : ot.Function
        A function.

    Returns
    -------
    firstOrderIndices : ot.Point(dimension)
        The first order Sobol' indices.
    totalIndices : ot.Point(dimension)
        The total order Sobol' indices.

    Reference
    ---------
    https://github.com/mbaudin47/otbenchmark/blob/master/otbenchmark/JanonSensitivityAlgorithm.py
    """

    def JanonKernel(outputSampleB, muB, outputSampleE, muE):
        muEB = (muE + muB) / 2.0
        yE_centered = outputSampleE - muEB
        yB_centered = outputSampleB - muEB
        numerator = sampleDotProduct(yE_centered, yB_centered)
        y_squared = (square(outputSampleE) + square(outputSampleB)) / 2.0 - muEB**2
        denominator = np.sum(y_squared)
        return numerator / denominator

    dim = inputSampleA.getDimension()
    # A
    outputSampleA = gFunction(inputSampleA)
    muA = outputSampleA.computeMean()[0]
    # B
    outputSampleB = gFunction(inputSampleB)
    muB = outputSampleB.computeMean()[0]
    firstOrderIndices = ot.Point(dim)
    totalIndices = ot.Point(dim)
    square = ot.SymbolicFunction(["x"], ["x ^ 2"])
    for i in range(dim):
        inputSampleE = ot.Sample(inputSampleA)
        inputSampleE[:, i] = inputSampleB[:, i]
        outputSampleE = gFunction(inputSampleE)
        muE = outputSampleE.computeMean()[0]
        firstOrderIndices[i] = JanonKernel(outputSampleB, muB, outputSampleE, muE)
        totalIndices[i] = 1.0 - JanonKernel(outputSampleA, muA, outputSampleE, muE)
    return firstOrderIndices, totalIndices


def computeDigits(expected, computed, basis=2.0):
    """
    Compute the number of base-b digits common in expected and computed.
    Compute the number of common base-b digits
    between expected and computed.
    Parameters
    ----------
    expected : float
        The expected value
    computed : float
        The computed value
    basis : float
        The basis
    d : float
        The number of common digits
    Examples
    --------
    >>> exact = 1.0
    >>> computed = 1.0
    >>> d = computeDigits(exact, computed)
    We can se the basis if required.
    >>> exact = 1.0
    >>> computed = 1.0
    >>> basis = 10.0
    >>> d = computeDigits(exact, computed, basis)
    """
    relerr = relativeError(expected, computed)
    dmin = 0
    dmax = -np.log(2 ** (-53)) / np.log(basis)
    if relerr == 0.0:
        d = dmax
    else:
        d = -np.log(relerr) / np.log(basis)
        d = max(dmin, d)
    return d


def relativeError(expected, computed):
    """
    Compute the relative error between expected and computed.
    Compute the relative error
    between expected and computed.
    If expected is zero, the relative error in infinite.
    Parameters
    ----------
    expected : float
        The expected value.
    computed : float
        The computed value.
    Examples
    --------
    >>> exact = 1.0
    >>> computed = 1.0
    >>> relerr = relativeError(exact, computed)
    """
    if (expected == 0.0) & (computed == 0):
        e = 0
    elif expected == 0.0:
        e = float("inf")
    else:
        e = abs(computed - expected) / abs(expected)
    return e


def print_indices_and_tuples(groups_sensitivity_values, groups_list):
    """
    Print the sensitivity indices and the groups.

    Parameters
    ----------
    groups_sensitivity_values : list(float)
        The sensitivity indices.
    groups_list : list(list(int))
        The list of groups of variables.

    Returns
    -------
    None.

    """
    if len(groups_list) != len(groups_sensitivity_values):
        raise ValueError(
            "The indices values have length %d, but the tuples have length %d"
            % (len(groups_sensitivity_values), len(groups_list))
        )
    print("+ Tuples and values :")
    for i in range(len(groups_sensitivity_values)):
        print(groups_list[i], ":", groups_sensitivity_values[i])
    return


def numpyArgsortSample(sample):
    """
    Returns the indices of the points in the Sample, in decreasing order.

    This is a workaround for OpenTURNS <= 1.20 where Sample.argsort()
    is not implemented.

    Parameters
    ----------
    sample : ot.Sample
        The sample.

    Returns
    -------
    indices : list(int)
        The indices in the ordered sample.

    """
    # Use the opposite to sort in decreasing order
    values_array = -np.array(sample)
    values_array = values_array.flatten()
    indices = values_array.argsort()
    indices = indices.flatten()
    return indices


def computeAllGroups(dimension):
    """
    Compute all tuples of a length lower or equal to dimension.

    Parameters
    ----------
    dimension : int
        The dimension.

    Returns
    -------
    groups_list : list(list(int))
        The list of groups.

    """
    # Compute groups
    groups_list = []
    stuff = list(range(dimension))
    first = True
    for L in range(len(stuff) + 1):
        for subset in itertools.combinations(stuff, L):
            # print(subset)
            if first:
                # The first one is empty
                first = False
                continue
            else:
                groups_list.append(subset)
    return groups_list


def computeNumberOfItemsByLine(total_nb_items):
    """
    Compute the number of represented items in a line.

    Parameters
    ----------
    total_nb_items : int
        The total number of values of values of plot.

    Returns
    -------
    items_by_column : list(number_of_lines)
        Each item in the list is the number of values in a line.
        The sum of values in items_by_column is equal to total_nb_items.
        The minimum number of values in a line is 1.

    """
    if total_nb_items == 1:
        items_by_column = [1]
    elif total_nb_items == 2:
        items_by_column = [2]
    else:
        number_of_lines = int(np.floor(math.sqrt(total_nb_items)))
        average_number_by_line = int(np.floor(total_nb_items / number_of_lines))
        items_by_column = [average_number_by_line for i in range(number_of_lines)]
        items_by_column[-1] = total_nb_items - average_number_by_line * (
            number_of_lines - 1
        )
    return items_by_column


def plotPCESensitivityTreemap(
    sensitivity,
    outputDescription,
    text_size=1.0,
    verbose=False,
    value_threshold=0.01,
    items_by_column=None,
):
    """
    Plot a sensitivity analysis Treemap.

    Parameters
    ----------
    sensitivity : ot.FunctionalChaosSobolIndices()
        The sensitivity analysis of a FunctionalChaos.
    outputDescription : ot.Description(outputDimension)
        The output description.
    text_size : float > 0, optional
        The text size. The default is 0.5.
    verbose : bool, optional
        If True, print intermediate messages. The default is False.
    value_threshold : float > 0, optional
        The Sobol' sensitivity indice threshold. The default is 0.01.
        Any interaction Sobol' indice lower than this threshold is ignored.
        A remainder set is created which represents the group of ignored
        interactions.
    items_by_column : list(int), optional.
        By default, applies an automatic rule
        The number of items in each line.

    Returns
    -------
    grid : ot.GridLayout(1, outputDimension)
        The plot.

    """
    fcresult = sensitivity.getFunctionalChaosResult()
    distribution = fcresult.getDistribution()
    inputDimension = distribution.getDimension()
    inputDescription = distribution.getDescription()
    metamodel = fcresult.getMetaModel()
    # Workaround for bug https://github.com/openturns/openturns/issues/2285
    # outputDescription = metamodel.getOutputDescription()
    outputDimension = metamodel.getOutputDimension()
    if verbose:
        print("Input description = ", inputDescription)
        print("Output description = ", outputDescription)

    grid = ot.GridLayout(1, outputDimension)
    for marginalIndex in range(outputDimension):
        # Compute all the indices
        groupsList, interaction_sobol_group_list = ComputeAllSobolInteractionIndices(
            fcresult
        )
        fullNumberOfGroups = len(groupsList)

        # Filter small indices
        groupsList_threshold = []
        interaction_sobol_group_list_threshold = []
        for index in range(len(groupsList)):
            if interaction_sobol_group_list[index] > value_threshold:
                groupsList_threshold.append(groupsList[index])
                interaction_sobol_group_list_threshold.append(
                    interaction_sobol_group_list[index]
                )

        # Create labels
        group_labels = []
        for group in groupsList_threshold:
            label = "[" + ",".join(inputDescription[group]) + "]"
            group_labels.append(label)

        # Add the remainder, if any
        if len(groupsList_threshold) < fullNumberOfGroups:
            remainder = 1.0 - sum(interaction_sobol_group_list_threshold)
            group_labels.append("*")
            interaction_sobol_group_list_threshold.append(remainder)

        if verbose:
            print("Sum of detected indices = ", sum(interaction_sobol_group_list))
            for i in range(len(groupsList_threshold)):
                print(
                    "groupsList = ",
                    groupsList_threshold[i],
                    "  groups_sensitivity_values = %.4f"
                    % (interaction_sobol_group_list_threshold[i]),
                    "  group_labels = ",
                    group_labels[i],
                )

        #
        n_groups = len(groupsList_threshold)
        print("n_groups = ", n_groups)
        treemap = tm.TreeMap(
            interaction_sobol_group_list_threshold,
            group_labels,
            text_size=text_size,
        )
        if items_by_column is None:
            number_of_columns = int(np.sqrt(n_groups))
            items_by_column = treemap.searchForBestNumberOfItemsByLine(
                number_of_columns
            )
        if verbose:
            print("items_by_column = ", items_by_column)
        graph = treemap.columnwise(items_by_column)
        graph.setTitle('%s, "*" = %.2e' % (outputDescription[marginalIndex], remainder))
        graph.setAxes(False)
        grid.setGraph(0, marginalIndex, graph)

    grid.setTitle("Interaction sensitivity indices")
    return grid


def multiBootstrap(*data):
    """
    Bootstrap multiple samples at once.

    Parameters
    ----------
    data : sequence of Sample
        Multiple samples to bootstrap.

    Returns
    -------
    data_boot : sequence of Sample
        The bootstrap samples.
    """
    assert len(data) > 0, "empty list"
    size = data[0].getSize()
    selection = ot.BootstrapExperiment.GenerateSelection(size, size)
    return [Z[selection] for Z in data]


def computeSparseLeastSquaresChaos(X, Y, basis, total_degree, distribution):
    """
    Create a sparse polynomial chaos with least squares.

    * Uses the enumeration rule from basis.
    * Uses LeastSquaresStrategy to compute the coefficients from
    linear least squares.
    * Uses LeastSquaresMetaModelSelectionFactory to select the polynomials
    in the basis using least angle regression stepwise (LARS)
    * Utilise LeastSquaresStrategy pour calculer les
    coefficients par moindres carrés.
    * Uses FixedStrategy to keep all coefficients that LARS has selected,
    up to the given maximum total degree.

    Parameters
    ----------
    X : Sample(n)
        The input training design of experiments with n points
    Y : Sample(n)
        The input training design of experiments with n points
    basis : Basis
        The multivariate orthogonal polynomial basis
    total_degree : int
        The maximum total polynomial degree
    distribution : Distribution
        The distribution of the input random vector

    Returns
    -------
    result : FunctionalChaosResult
        The polynomial chaos result
    """
    selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
    projectionStrategy = ot.LeastSquaresStrategy(selectionAlgorithm)
    enumerateFunction = basis.getEnumerateFunction()
    if ot.__version__ == "1.18":
        idx = enumerateFunction.getMaximumDegreeStrataIndex(total_degree)
        basisSize = enumerateFunction.getStrataCumulatedCardinal(idx)
    else:
        basisSize = enumerateFunction.getBasisSizeFromTotalDegree(total_degree)
    adaptiveStrategy = ot.FixedStrategy(basis, basisSize)
    algo = ot.FunctionalChaosAlgorithm(
        X, Y, distribution, adaptiveStrategy, projectionStrategy
    )
    algo.run()
    result = algo.getResult()
    return result


def computeChaosSensitivity(X, Y, basis, total_degree, distribution):
    """
    Compute the first and total order Sobol' indices from a polynomial chaos.

    Parameters
    ----------
    X : ot.Sample
        Input design
    Y : ot.Sample
        Output design
    basis : Basis
        Tensorized basis
    total_degree : int
        Maximal total degree
    distribution : Distribution
        Input distribution

    Returns
    -------
    first_order, total_order: list of float
        The first and total order indices.
    """
    dim_input = X.getDimension()
    result = computeSparseLeastSquaresChaos(X, Y, basis, total_degree, distribution)
    chaosSI = ot.FunctionalChaosSobolIndices(result)
    first_order = [chaosSI.getSobolIndex(i) for i in range(dim_input)]
    total_order = [chaosSI.getSobolTotalIndex(i) for i in range(dim_input)]
    return first_order, total_order


def computeBootstrapChaosSobolIndices(
    X, Y, basis, total_degree, distribution, bootstrap_size
):
    """
    Computes a bootstrap sample of first and total order indices from polynomial chaos.

    Parameters
    ----------
    X : ot.Sample
        Input design
    Y : ot.Sample
        Output design
    basis : Basis
        Tensorized basis
    total_degree : int
        Maximal total degree
    distribution : Distribution
        Input distribution
    bootstrap_size : interval
        The bootstrap sample size

    Returns
    ----------
    fo_sample: ot.Sample(n, dim_input)
        The first order indices
    to_sample: ot.Sample(n, dim_input)
        The total order indices
    """
    dim_input = X.getDimension()
    fo_sample = ot.Sample(0, dim_input)
    to_sample = ot.Sample(0, dim_input)
    unit_eps = ot.Interval([1e-9] * dim_input, [1 - 1e-9] * dim_input)
    for i in range(bootstrap_size):
        X_boot, Y_boot = multiBootstrap(X, Y)
        first_order, total_order = computeChaosSensitivity(
            X_boot, Y_boot, basis, total_degree, distribution
        )
        if unit_eps.contains(first_order) and unit_eps.contains(total_order):
            fo_sample.add(first_order)
            to_sample.add(total_order)
    return fo_sample, to_sample


def computeSobolIndicesConfidenceInterval(fo_sample, to_sample, alpha=0.95):
    """
    From a sample of first or total order indices,
    compute a bilateral confidence interval of level alpha.

    Estimates the distribution of the first and total order Sobol' indices
    from a bootstrap estimation.
    Then computes a bilateral confidence interval for each marginal.

    Parameters
    ----------
    fo_sample: ot.Sample(n, dim_input)
        The first order indices
    to_sample: ot.Sample(n, dim_input)
        The total order indices
    alpha : float
        The confidence level

    Returns
    -------
    fo_interval : ot.Interval
        The confidence interval of first order Sobol' indices
    to_interval : ot.Interval
        The confidence interval of total order Sobol' indices
    """
    dim_input = fo_sample.getDimension()
    fo_lb = [0] * dim_input
    fo_ub = [0] * dim_input
    to_lb = [0] * dim_input
    to_ub = [0] * dim_input
    for i in range(dim_input):
        fo_i = fo_sample[:, i]
        to_i = to_sample[:, i]
        beta = (1.0 - alpha) / 2
        fo_lb[i] = fo_i.computeQuantile(beta)[0]
        fo_ub[i] = fo_i.computeQuantile(1.0 - beta)[0]
        to_lb[i] = to_i.computeQuantile(beta)[0]
        to_ub[i] = to_i.computeQuantile(1.0 - beta)[0]

    # Create intervals
    fo_interval = ot.Interval(fo_lb, fo_ub)
    to_interval = ot.Interval(to_lb, to_ub)
    return fo_interval, to_interval


def computeAndDrawSobolIndices(
    X, Y, basis, total_degree, distribution, bootstrap_size=500, alpha=0.95
):
    """
    Compute and draw Sobol' indices from a polynomial chaos

    Compute the PCE based on a given sample size.
    Compute confidence intervals at level alpha from bootstrap.

    Parameters
    ----------
    X : ot.Sample
        Input design
    Y : ot.Sample
        Output design
    basis : ot.Basis
        Tensorized basis
    total_degree : int
        Maximal total degree
    distribution : ot.Distribution
        Input distribution
    bootstrap_size : int
        The bootstrap sample size

    Returns
    -------
    graph : ot.Graph
        The Sobol' indices.
    """
    fo_sample, to_sample = computeBootstrapChaosSobolIndices(
        X, Y, basis, total_degree, distribution, bootstrap_size
    )

    fo_interval, to_interval = computeSobolIndicesConfidenceInterval(
        fo_sample, to_sample, alpha
    )
    input_names = distribution.getDescription()
    graph = ot.SobolIndicesAlgorithm.DrawSobolIndices(
        input_names,
        fo_sample.computeMean(),
        to_sample.computeMean(),
        fo_interval,
        to_interval,
    )
    n = X.getSize()
    graph.setTitle(f"Sobol' indices - n={n}")
    # graph.setIntegerXTick(True)
    return graph


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


def PrintSensitivityGroupMarkdown(
    groups_sensitivity_values, group_labels, sort_table=True
):
    """
    Print Sobol' sensitivity indices with Markdown format.

    Parameters
    ----------
    groups_sensitivity_values : list(float)
        The Sobol' interaction index of each corresponding group of
        variables in group_list.

    group_labels : list(str)
        The labels of each group of variables.

    sort_table : bool
        If True, then print the indices in decreasing order.
        Set to False to print an unsorted table.

    """
    number_of_groups = len(groups_sensitivity_values)
    if len(group_labels) != number_of_groups:
        raise ValueError(
            f"The number of labels is {len(group_labels)} "
            f"but the number of sensitivity values is {number_of_groups}"
        )

    if sort_table:
        # Sort sensitivity indices
        groups_sensitivity_values_sample = ot.Sample.BuildFromPoint(
            groups_sensitivity_values
        )
        indices = numpyArgsortSample(groups_sensitivity_values_sample)
        groups_sensitivity_values = [groups_sensitivity_values[i] for i in indices]
        group_labels = [group_labels[i] for i in indices]

    # Print found groups labels
    print("| Group | Interaction sensitivity index |")
    print("|---|---|")
    for i in range(number_of_groups):
        print(f"| {group_labels[i]} | {groups_sensitivity_values[i]:.4f} |")
    print(f"| Sum | {sum(groups_sensitivity_values):.4f}|")
    return


def ComputeSobolInteractionIndicesByOutput(polynomialChaosResult, marginalIndex=0):
    """
    Compute Sobol' interaction indices of a PCE

    Parameters
    ----------
    polynomialChaosResult : ot.FunctionalChaosResult
        The polynomial chaos expansion.
    marginalIndex : int
        The marginal index of the output.

    Returns
    -------
    group_list : list(list(int))
        The list of groups of variables.
        Each group of variables is a list of indices representing
        the index of each variable in the group.

    interaction_sobol_group_list : list(float)
        The Sobol' interaction index of each corresponding group of
        variables in group_list.
    """
    # Get the part of variance of each multi-index
    functionalChaosSobolIndices = ot.FunctionalChaosSobolIndices(polynomialChaosResult)
    input_dimension = polynomialChaosResult.getInputSample().getDimension()
    part_of_variances = functionalChaosSobolIndices.getPartOfVariance(marginalIndex)
    number_of_coefficients = part_of_variances.getDimension()
    orthogonalBasis = polynomialChaosResult.getOrthogonalBasis()
    enumerateFunction = orthogonalBasis.getEnumerateFunction()
    indices = polynomialChaosResult.getIndices()

    # Create the groups
    group_dictionnary = defaultdict(float)
    for i in range(number_of_coefficients):
        multi_indices = enumerateFunction(indices[i])
        if sum(multi_indices) > 0:
            # Compute the group of input active variable of each multi-index
            group_of_variables = [j for j in range(input_dimension) if multi_indices[j] > 0]
            group_of_variables = tuple(group_of_variables)
            group_dictionnary[group_of_variables] += part_of_variances[i]

    # Merge the groups
    group_list = sorted(group_dictionnary)
    part_of_variance_list = [group_dictionnary[key] for key in group_list]

    # Sort the groups by decreasing part of variance
    part_of_variance_array = np.array(part_of_variance_list)
    group_indices = (-part_of_variance_array).argsort()
    group_list = [group_list[i] for i in group_indices]
    interaction_sobol_group_list = [part_of_variance_list[i] for i in group_indices]
    return group_list, interaction_sobol_group_list

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
