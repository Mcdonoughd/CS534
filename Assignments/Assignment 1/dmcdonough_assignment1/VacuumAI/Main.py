#Vacuum Reflex Agent By Daniel Mcdonough 9/16

import os
import os.path
import math
from numpy import genfromtxt
import collections

#The Environment OBJ
class Environment(object):
    def __init__(self):

        #init environment from file for maximum modularity
        filename = input('Please Enter Environment File: ')        # instantiate locations and conditions

        #check if file exists
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            print("File exists and is readable")
        else:
            print("Either the file is missing or not readable")
            exit(1)

        #set location conditions to a field
        self.locationCondition = genfromtxt(filename, delimiter=',')
        #print(len(self.locationCondition.shape))

        self.width = self.locationCondition.shape[0]-1 #0 indexed

        #check if height is a factor...
        if len(self.locationCondition.shape) == 1:
            self.height = 0
        else:
            self.height = self.locationCondition.shape[1]-1 #0 indexed


#Reflex Agent
class SimpleReflexVacuumAgent(Environment):
    def __init__(self, Environment):
        #print init Environment
        print(Environment.locationCondition)

        # Instantiate performance measurement
        self.Score = 0

        # place vacuum at a given location

        #input X coord
        self.initlocationX = input('Enter the Starting location index (X):')

        #check valid input
        if self.initlocationX.isdigit():
            self.initlocationX = int(self.initlocationX)
        else:
            print("INPUT NOT A DIGIT")
            exit(1)

        if not self.initlocationX > 0 and not self.initlocationX <= Environment.width:
            print("Given Location is OOB")
            exit(1)

        #check if NxN Case
        if len(Environment.locationCondition.shape)>1:
            #input Y Coord
            self.initlocationY = input('Enter the Starting location index (Y):')
            if self.initlocationY.isdigit():
                self.initlocationY = int(self.initlocationY)
            else:
                print("INPUT NOT A DIGIT")
                exit(1)

            #check if givin location is valid


            if not self.initlocationY > 0 and not self.initlocationY <= Environment.height:
                print("Given Location is OOB")
                exit(1)
        #otherwise N is always 0
        else:
            self.initlocationY = 0

        #Check if location is an actual room
        if len(Environment.locationCondition.shape) == 1:
            if math.isnan(Environment.locationCondition[self.initlocationX]):
                print("ERROR, Location is not a room")
                exit(1)
            else:
                print(Environment.locationCondition[self.initlocationX])

        else:
            if math.isnan(Environment.locationCondition[self.initlocationY,self.initlocationX]):
                print("ERROR, Location is not a room")
                exit(1)
            else:
                print(Environment.locationCondition[self.initlocationY,self.initlocationX])


        # we only need to check the X axis as the vacuum can only move left or right
        self.vacuumLocation = self.initlocationX


        #This is the "vision" the AI has of rooms it knows has dirt/has visited
        # +/- "ghost rooms" are added as a buffer, to not clean one room then end
        # 1 = not visited
        # 0 = visited


        self.vision = collections.OrderedDict({ self.initlocationX: 1})

        # NOTE: Order of this actually matters, but determining which order would be best would be assuming that the AI knows it's location in regard to the environment (which it should not)
        self.vision[self.initlocationX - 1] = 1
        self.vision[self.initlocationX + 1] = 1



        #go for 1000 steps or until clean
        for step in range(1000):
           # print("Current Location: ",self.vacuumLocation)
            #print("Performance Measurement: " + str(self.Score))
            #check current location

           #1XN CASE
            if len(Environment.locationCondition.shape) == 1:
                #suck if dirty
                if Environment.locationCondition[self.vacuumLocation] > 0:
                    self.suck(Environment)
                else:
                    # otherwise check vision...
                    self.checkVis(Environment)

            #NXN CASE
            else:
                if Environment.locationCondition[self.initlocationY,self.vacuumLocation] > 0:
                    self.suck(Environment)
                else:
                    #otherwise check vision...
                    self.checkVis(Environment)

            # check if goal state is reached
            if self.checkGoal():
                print("Everything Should Be Clean")
                print(Environment.locationCondition)
                print("Performance Measurement: " + str(self.Score))
                return
        print("Ran for 1000 Steps")
        print(Environment.locationCondition)
        print("Performance Measurement: " + str(self.Score))
        return



    #suck the dirt on current square
    def suck(self,Environment):
        print("Location ", self.vacuumLocation," is Dirty." )
        # SUCC
        if len(Environment.locationCondition.shape) == 1:
            Environment.locationCondition[self.vacuumLocation] = 0
            self.Score += 1  # update score
            self.vision[self.vacuumLocation] = 0  # update vision
            print("Location ", self.vacuumLocation, " has been Cleaned.")
        else:
            Environment.locationCondition[self.initlocationY, self.vacuumLocation] = 0
            self.Score += 1 #update score
            self.vision[self.vacuumLocation] = 0  # update vision
            print("Location ", self.vacuumLocation," has been Cleaned.")

    #move left
    def left(self,Environment):
        #check if going left is possible
        if len(Environment.locationCondition.shape) == 1:
            if self.vacuumLocation <= 0 or math.isnan(Environment.locationCondition[self.vacuumLocation-1]):
                self.vision[self.vacuumLocation - 1] = 0  # there is not another room
                self.vision[self.vacuumLocation] = 0
                print("CANNOT MOVE LEFT, DOING NOTHING.")
            else:
                #move left
                self.Score -= 1
                self.vacuumLocation -= 1
                self.vision[self.vacuumLocation] = 0  # update vision
                self.vision[self.vacuumLocation-1] = 1 #is there another room?
                print("Moved Left")
        else:
            if self.vacuumLocation <= 0 or math.isnan(Environment.locationCondition[self.initlocationY,self.vacuumLocation-1]):
                self.vision[self.vacuumLocation - 1] = 0  # there is not another room
                self.vision[self.vacuumLocation] = 0
                print("CANNOT MOVE LEFT, DOING NOTHING.")
            else:
                #move left
                self.Score -= 1
                self.vacuumLocation -= 1
                self.vision[self.vacuumLocation] = 0  # update vision
                self.vision[self.vacuumLocation-1] = 1 #is there another room?
                print("Moved Left")


    #move right
    def right(self,Environment):
        #check if going right is possible
        if len(Environment.locationCondition.shape) == 1:
            if self.vacuumLocation >= Environment.width or math.isnan(Environment.locationCondition[self.vacuumLocation + 1]):
                self.vision[self.vacuumLocation] = 0  # There is not another room
                self.vision[self.vacuumLocation+1] = 0
                #self.rightwall = 1
                print("CANNOT MOVE RIGHT, DOING NOTHING.")
            else:
                # move right
                self.Score -= 1
                self.vacuumLocation += 1
                self.vision[self.vacuumLocation] = 0  # update vision
                self.vision[self.vacuumLocation + 1] = 1  # is there another room?
                print("Moved Right")
        else:
            if self.vacuumLocation >= Environment.width or math.isnan(Environment.locationCondition[self.initlocationY,self.vacuumLocation+1]):
                self.vision[self.vacuumLocation + 1] = 0  # There is not another room
                self.vision[self.vacuumLocation] = 0
                print("CANNOT MOVE RIGHT, DOING NOTHING.")
            else:
                #move right
                self.Score -= 1
                self.vacuumLocation += 1
                self.vision[self.vacuumLocation] = 0  # update vision
                self.vision[self.vacuumLocation+1] = 1 #is there another room?
                print("Moved Right")


    #check if goal state is reached (aka are all entries in the vision clean?)
    def checkGoal(self):
        for key, value in self.vision.items():
            #print("Location: ",key,", Viewed: ", value)
            if value == 1:
                return False
        return True


    def checkVis(self,Environment):
        #if there are any rooms not viewed then move in that direction...
        for key, value in self.vision.items():
            if value == 1:
                if self.vacuumLocation > key:
                    self.left(Environment)
                    return
                else:
                    self.right(Environment)
                    return



theEnvironment = Environment()
theVacuum = SimpleReflexVacuumAgent(theEnvironment)