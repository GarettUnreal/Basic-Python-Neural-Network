import numpy
import math
import sys
from reprint import output

def sigmoid(x):
    return 1 / ( 1 + math.exp(-x))

def sigmoidPrime(x):
    sig = sigmoid(x)
    return sig * ( 1 - sig )

class NeuralNetwork:

    # @param {Array} nodesPerLayer An array containing the
    # number of nodes per layer. The first layer is at index
    # 0 and the last layer is at the last index.
    # The value at each index contains the number of nodes
    # for that layer.
    def __init__(self, nodesPerLayer):

        # The mathematical notation does not easily jusfity
        # an index at zero.
        self.weights = [0]
        self.betas = [0]

        for index in range( len( nodesPerLayer ) - 1 ):

            nodesNext = nodesPerLayer[ index + 1 ]
            nodesCurrent = nodesPerLayer[ index ]

            self.weights.append( 1 * numpy.random.rand(nodesCurrent,nodesNext) - 0.5 )
            self.betas.append( 0.2 * numpy.random.rand(nodesNext) - 0.1 )

        # The following are temporary variables to help
        # speed up computation
        self.xvalues = [0]
        self.sigmaPrimeXvalues = [0]
        self.Q = [0]

        self.alphas = []

        for layer in xrange( len( nodesPerLayer ) - 1 ):
            
            nodesNext = nodesPerLayer[ layer + 1 ]
            nodesCurrent = nodesPerLayer[ layer ]

            self.xvalues.append( numpy.zeros(nodesNext) )
            self.sigmaPrimeXvalues.append( numpy.zeros(nodesNext) )
            self.alphas.append( numpy.zeros(nodesCurrent) )
            self.Q.append( numpy.identity( nodesNext ))

        self.alphas.append( numpy.zeros( nodesPerLayer[ len( nodesPerLayer ) - 1 ] ) ) 

        self.sigmoidVec = numpy.vectorize(sigmoid)
        self.sigmoidPrimVec = numpy.vectorize(sigmoidPrime)

        self.nodesPerLayer = nodesPerLayer

    def predict( self, inputVector ):
        alpha = inputVector
        for layer in xrange( 1, len( self.xvalues ) ):
            xValues = numpy.transpose( self.weights[layer] ).dot( alpha ) + self.betas[layer]
            alpha = self.sigmoidVec( xValues )
        return alpha

    def learn( self, inputVectors, outputVectors ):

        previousParameters = {
            "weights": self.weights,
            "betas": self.betas
        }

        previousDerivatives = None

        precision = 0.000001
        gamma = 0.1

        stepDifference = 100
        previousCost = self.__calculateCost( inputVectors, outputVectors )

        iterationCount = 1
        while True:

            print( "Calculating Derivatives" )
            print( "" )
            derivatives = self.__calcDerivativesMultiple( inputVectors, outputVectors )

            newParameters = self.__updateParametersGradDesc( derivatives, gamma )
            self.weights = newParameters[ "weights" ]
            self.betas = newParameters[ "betas" ]

            #if previousDerivatives is not None:
            #    gamma = self.__updateStepSize( derivatives, previousDerivatives, newParameters, previousParameters )

            stepDifference = self.__calcDifference( newParameters, previousParameters )
            cost = self.__calculateCost( inputVectors, outputVectors )

            print( "Iteration ", iterationCount, " Step Difference: ", stepDifference, "Average Cost: ", cost )
            print( "Cost Difference ", ( cost - previousCost ) )
            print( "Gamme ", gamma )
            if abs( cost - previousCost ) < precision:
                break

            previousCost = cost
            previousParameters = newParameters
            iterationCount = iterationCount + 1
            previousDerivatives = derivatives

    def __updateStepSize( self, derivatives, previousDerivatives, newParameters, previousParameters ):
        
        diffDerivsVec = self.__vectorizeDiffDerivs( derivatives, previousDerivatives )
        diffParamsVec = self.__vectorizeDiffParams( newParameters, previousParameters )

        return diffParamsVec.dot( diffDerivsVec ) / diffDerivsVec.dot( diffDerivsVec )

    def __vectorizeDiffDerivs( self, derivatives, previousDerivatives ):
        weightDerivatives = derivatives["weightDerivatives"]
        betaDerivatives = derivatives["betaDerivatives"]
        weightDerivativesPrev = previousDerivatives["weightDerivatives"]
        betaDerivativesPrev = previousDerivatives["betaDerivatives"]
        result = []
        for layer in xrange(1, len( weightDerivatives ) ):
            result = numpy.append( result, weightDerivatives[layer] - weightDerivativesPrev[layer] )
            result = numpy.append( result, betaDerivatives[layer] - betaDerivativesPrev[layer] )
        return result

    def __vectorizeDiffParams( self, newParameters, previousParameters ):
        weights = newParameters[ "weights" ]
        betas = newParameters[ "betas" ]
        weightsPrev = previousParameters[ "weights" ]
        betasPrev = previousParameters[ "betas" ]
        result = []
        for layer in xrange(1, len( weights ) ):
            result = numpy.append( result, weights[layer] - weightsPrev[layer] )
            result = numpy.append( result, betas[layer] - betasPrev[layer] )
        return result

    def __calculateCost( self, inputVectors, outputVectors ):
        cost = 0
        for index in xrange( len( inputVectors ) ):
            diff = self.predict( inputVectors[index] ) - outputVectors[index]
            cost += numpy.linalg.norm( diff )
        return cost / len( inputVectors )

    def __updateParametersGradDesc( self, derivatives, gamma ):

        newWeights = [0]
        newBetas = [0]
        weightDerivatives = derivatives["weightDerivatives"]
        betaDerivatives = derivatives["betaDerivatives"]

        for layer in xrange( 1, len( weightDerivatives ) ):
            newWeights.append( self.weights[layer] - gamma * weightDerivatives[layer] )
            newBetas.append( self.betas[layer] - gamma * betaDerivatives[layer] )

        return {
            "weights": newWeights,
            "betas": newBetas
        }

    def __calcDifference( self, newParameters, previousParameters ):
        difference = 0
        weights = newParameters["weights"]
        betas = newParameters["betas"]
        previousWeights = previousParameters["weights"]
        previousBetas = previousParameters["betas"]
        for layer in xrange( len( weights ) ):
            difference = difference + numpy.linalg.norm( weights[layer] - previousWeights[layer] )
            difference = difference + numpy.linalg.norm( betas[layer] - previousBetas[layer] )
        return difference
    
    def __updateFastCalculationData( self, inputVector ):
        self.alphas[0] = inputVector
        for layer in xrange( 1, len( self.xvalues ) ):
            self.xvalues[layer] = numpy.transpose( self.weights[layer] ).dot( self.alphas[layer-1] ) + \
                self.betas[layer]
            self.alphas[layer] = self.sigmoidVec( self.xvalues[layer] )
            self.sigmaPrimeXvalues[layer] = self.sigmoidPrimVec( self.xvalues[layer] )

        for layer in xrange( len( self.Q ) - 2, 0, -1 ):

            self.Q[layer] = self.Q[layer+1].dot(
                    numpy.transpose(
                        numpy.multiply( self.weights[layer+1], self.sigmaPrimeXvalues[layer+1] )
                    )
                )
    
    #
    # Calculate the derivatives with respect to the loss function of a single
    # input.
    # @param {Vector} inputVector The input vectors of the training data.
    # @param {Vector} outputVector The actual corresponding output vectors for each input
    # vector of the training data.
    def __calcDerivatives( self, inputVector, outputVector ):

        self.__updateFastCalculationData( inputVector )
        depth = len( self.nodesPerLayer )

        diff = self.alphas[depth-1] - outputVector
        diffT = numpy.transpose( diff )

        weightDerivatives = [0]
        betaDerivatives = [0]

        for layer in xrange( 1, depth ):

            weightDimensions = self.weights[layer].shape
            rows = weightDimensions[0]
            columns = weightDimensions[1]

            R = numpy.diag( self.sigmaPrimeXvalues[layer] )
            weightGradient = numpy.outer( self.alphas[layer-1], diffT ).dot( self.Q[layer] ).dot( R )
            betaGradient = R.dot( numpy.transpose(self.Q[layer]) ).dot( diff )

            weightDerivatives.append( weightGradient )
            betaDerivatives.append( betaGradient )

        return {
            "weightDerivatives": weightDerivatives,
            "betaDerivatives": betaDerivatives
        }

    def __calcDerivativesMultiple( self, inputVectors, outputVectors ):

        derivatives = self.__calcDerivatives( inputVectors[0], outputVectors[0] )

        temp = 0
        for index in xrange( 1, len( inputVectors ) ):
            
            if temp == 100:
                information = "Derivative calculation for training data " + \
                    "{:3.4f}".format( 100.0 * float(index) / float(len(inputVectors)) ) + "% "
                #print( information )
                temp = 0
            temp = temp + 1
            
            derivatives = self.__addDerivatives(
                derivatives,
                self.__calcDerivatives( inputVectors[index], outputVectors[index] )
            )

        return derivatives

    def __addDerivatives( self, derivativesOne, derivativesTwo ):
        resultWeights = [0]
        resultBetas = [0]
        for layer in xrange( 1, len( derivativesOne["weightDerivatives"] ) ):
            resultWeights.append( derivativesOne["weightDerivatives"][layer] + \
                derivativesTwo["weightDerivatives"][layer] )
            resultBetas.append( derivativesOne["betaDerivatives"][layer] + \
                derivativesTwo["betaDerivatives"][layer] )
        return {
            "weightDerivatives": resultWeights,
            "betaDerivatives": resultBetas
        }

        

        
        
