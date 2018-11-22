from PIL import Image
import numpy
from NeuralNetwork import NeuralNetwork

def showMatrixAsImage( matrix ):
    matrixMax = numpy.max( matrix )
    matrixMin = numpy.min( matrix )
    normalizedMatrix = ( matrix + matrixMin * numpy.ones(matrix.shape) ) / ( matrixMax - matrixMin )
    formattedMatrix = ( 255 * normalizedMatrix ).astype('uint8')
    Image.fromarray(formattedMatrix).show()

#matrixString = '0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000'
#data = numpy.fromstring(matrixString,dtype=float,sep=' ')
#showMatrixAsImage( data.reshape((16,16)) )

dataFileName = "test\digitRecognitionData.txt"
fileContent = None
with open(dataFileName) as dataFile:
    fileContent = dataFile.readlines()
    dataFile.close()

inputVectors = []
outputVectors = []
'''
for lineIndex in xrange( len( fileContent ) ):
    line = fileContent[lineIndex]
    floats = [float(x) for x in line.split()]
    imageFloats = floats[:256]
    classFloats = floats[256:]
    imageFloatStrings = [ '{:.2f}'.format(x) for x in imageFloats ]
    classFloatStrings = [ '{:.2f}'.format(x) for x in classFloats ]
    inputVectors.append( numpy.fromstring( " ".join( imageFloatStrings ), dtype=float, sep=' ' ) )
    outputVectors.append( numpy.fromstring( " ".join( classFloatStrings ), dtype=float, sep=' ' ) )

nodesPerLayer = [256,30,25,20,15,10]
'''
inputVectors.append(numpy.fromstring('0 0 1',dtype=float,sep=' '))
outputVectors.append(numpy.fromstring('1 0 0',dtype=float,sep=' '))
inputVectors.append(numpy.fromstring('1 0 0',dtype=float,sep=' '))
outputVectors.append(numpy.fromstring('0 1 0',dtype=float,sep=' '))
inputVectors.append(numpy.fromstring('1 0 1',dtype=float,sep=' '))
outputVectors.append(numpy.fromstring('0 0 1',dtype=float,sep=' '))
nodesPerLayer = [3,4,3]

neuralNetwork = NeuralNetwork(nodesPerLayer)
neuralNetwork.learn(inputVectors,outputVectors)

prediction = neuralNetwork.predict(inputVectors[0])
print( "Prediction for the first input: " )
print(prediction)

prediction = neuralNetwork.predict(inputVectors[1])

print( "Prediction for the second input: " )
print(prediction)

prediction = neuralNetwork.predict(inputVectors[2])

print( "Prediction for the third input: " )
print(prediction)

