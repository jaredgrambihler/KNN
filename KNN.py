from mnist import MNIST
import time
import numpy as np

def importTrainData(dir = './mnist'):
    try:
        mndata = MNIST(dir)
        trainImages, trainLabels = mndata.load_training()
        return (trainImages, trainLabels)
    except FileNotFoundError:
        return None

def importTestData(dir = './mnist'):
    try:
        mndata = MNIST(dir)
        testImages, testLabels = mndata.load_testing()
        return (testImages, testLabels)
    except FileNotFoundError:
        return None

class KNN():

    def __init__(self):
        pass
    
    def train(self, trainData):
        #trainData is a tuple
        self.data = trainData

    def distance(self, image1, image2):
        #euclidian distance
        distance = 0
        distance = np.sum((np.array(image1) - np.array(image2))**2)
        return np.sqrt(distance)

    def manhattanDistance(self, image1, image2):
        #manhattan distance
        distance = np.abs(np.sum(np.array(image1) - np.array(image2)))
        return distance

    def initNeighbors(self, image, k, getDistance):
        #inits sorted list
        nearestNeighbors = []
        for i in range(k):
            localNeighbor = (self.data[1][i], getDistance(
                self.data[0][i], image) )#(label, distance)
            if len(nearestNeighbors) == 0:
                nearestNeighbors.append(localNeighbor)
            else:
                index  = 0 
                while index<len(nearestNeighbors):
                    if localNeighbor[1] < nearestNeighbors[index][1]: #compares distance
                        nearestNeighbors.insert(index, localNeighbor)
                        index += k
                    else:
                        index += 1
                if index == len(nearestNeighbors):
                    nearestNeighbors.insert(index, localNeighbor)
        return nearestNeighbors
        
    def classify(self, image, k, distance = 'Euclidian'):
    #classify an image for k nearest neighbors
    #returns the index of the label value
        #sets distance type
        if(distance == 'Euclidian'):
            getDistance = self.distance
        elif(distance == 'Manhattan'):
            getDistance = self.manhattanDistance

        #creates a list of sorted neighbors of length k
        nearestNeighbors = self.initNeighbors(image, k, getDistance)

        #goes through each image in the training data
        #if an image is closer than the current nearest neighbors,
        #that image is added to the nearest neighbors and the largest
        #neighbor is then removed
        for i in range(k, len(self.data[0])):
            distance = getDistance(self.data[0][i], image)
            index = 0
            while index < k:
                if(distance < nearestNeighbors[index][1]):
                    nearestNeighbors.insert(index, (self.data[1][i], distance) )
                    nearestNeighbors = nearestNeighbors[:-1]
                    index += k
                else:
                    index += 1
        
        labelCounts = [0 for x in range(10)]
        for neighbor in nearestNeighbors:
            labelNum = neighbor[0]
            labelCounts[labelNum] += 1
        return labelCounts.index(max(labelCounts))
        
    def test(self, testData, k, distance = 'Manhattan'):
        #test accuracy for k nearest neighbors
        accuracy = 0
        for i in range(len(testData[0])):
            label = self.classify(testData[0][i], k, distance)
            print('Classified Images: ' + str(i+1) + '/' + str(len(testData[0])))
            if label == testData[1][i]:
                accuracy += 1
        accuracy /= len(testData)
        return accuracy

def testAccuracy(classifier, testData, k, distance):
    #tests the accuracy of the nearest neighbors classifier
    startTime = time.perf_counter()
    accuracy = classifier.test(testData, k, distance)
    endTime = time.perf_counter()
    print('Accuracy Time :', endTime - startTime)
    print('KNN accuracy with', distance, 'distance and', k, 'nearest neighbors:', accuracy)

 
def main():
    trainData = importTrainData()
    testData = importTestData()

    #short data to save time
    shortTrainData = (trainData[0][:1000], trainData[1][:1000])
    shortTestData = (testData[0][:150], testData[1][:150])
    
    classifier = KNN()
    classifier.train(trainData)
    
    k = 5
    testAccuracy(classifier, shortTestData, k, distance = 'Euclidian')

if __name__ == '__main__':
    main()
