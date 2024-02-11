# -*- coding: utf-8 -*-
import openturns as ot

def ComputeSparseLeastSquaresChaos(
    inputTrain, outputTrain, multivariateBasis, totalDegree, distribution
):
    """
      Create a sparse polynomial chaos based on least squares.

      * Uses the enumerate rule in basis.
      * Uses the LeastSquaresStrategy to compute the coefficients based on
        least squares.
      * Uses LeastSquaresMetaModelSelectionFactory to use the LARS selection method.
      * Uses FixedStrategy in order to keep all the coefficients that the
        LARS method selected.

    Source : https://openturns.github.io/openturns/latest/auto_meta_modeling/polynomial_chaos_metamodel/plot_chaos_cv.html
    with a bug fix: replace getStrataCumulatedCardinal with getBasisSizeFromTotalDegree

      Parameters
      ----------
      inputTrain : Sample
          The input design of experiments.
      outputTrain : Sample
          The output design of experiments.
      multivariateBasis : Basis
          The multivariate chaos basis.
      totalDegree : int
          The total degree of the chaos polynomial.
      distribution : Distribution.
          The distribution of the input variable.

      Returns
      -------
      result : PolynomialChaosResult
          The estimated polynomial chaos.
    """
    selectionAlgorithm = ot.LeastSquaresMetaModelSelectionFactory()
    projectionStrategy = ot.LeastSquaresStrategy(
        inputTrain, outputTrain, selectionAlgorithm
    )
    enumerateFunction = multivariateBasis.getEnumerateFunction()
    basisSize = enumerateFunction.getBasisSizeFromTotalDegree(totalDegree)  # OK
    adaptiveStrategy = ot.FixedStrategy(multivariateBasis, basisSize)
    chaosalgo = ot.FunctionalChaosAlgorithm(
        inputTrain, outputTrain, distribution, adaptiveStrategy, projectionStrategy
    )
    chaosalgo.run()
    result = chaosalgo.getResult()
    return result



def compute_sample_Q2_q_norm(inputSample, outputSample, distribution, numberAttempts, q_norm, degree, split_fraction = 0.75):
    """
    For a given sample size N, for different q-norms,
    repeat the following experiment numberAttempts times:
    create a sparse least squares chaos and compute the Q2
    using n_valid points.
    """
    sampleSize = inputSample.getSize()
    mixingDistribution = ot.KPermutationsDistribution(sampleSize, sampleSize)
    Q2sample = ot.Sample(numberAttempts, len(q_norm))
    
    #Fix the degree
    totalDegree = degree
    
    for k in range(len(q_norm)):
        q = q_norm[k]
        # use of hyperbolic enumeration rule
        enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(inputSample.getDimension(), q)
        multivariateBasis = ot.OrthogonalProductPolynomialFactory([distribution.getMarginal(i) for i in range(distribution.getDimension())], enumerateFunction
)
        print("q-norm = {}".format(q))
        for i in range(numberAttempts):
            # Randomize the sample
            X_train = ot.Sample(inputSample)
            Y_train = ot.Sample(outputSample)
            newIndices = mixingDistribution.getRealization()
            X_train = X_train[newIndices]
            Y_train = Y_train[newIndices]
            # Split
            split_index = int(split_fraction * sampleSize)
            X_test = X_train.split(split_index)
            Y_test = Y_train.split(split_index)
            # Train
            chaosResult = ComputeSparseLeastSquaresChaos(
                X_train, Y_train, multivariateBasis, totalDegree, distribution
            )
            metamodel = chaosResult.getMetaModel()
            # Test
            val = ot.MetaModelValidation(X_test, Y_test, metamodel)
            Q2_score = val.computePredictivityFactor()
            Q2sample[i, k - 1] = Q2_score.norm1() / Q2_score.getDimension()
    return Q2sample