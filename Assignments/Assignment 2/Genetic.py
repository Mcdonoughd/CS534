#Genetic Implementation for TSP by Daniel McDonough 9/30

import sys, random
from math import sqrt
import os
from numpy import genfromtxt


#cartesian_matrix from coords
def cartesian_matrix(co):
    matrix = {}
    for i, (x1, y1) in enumerate(co):
        for j, (x2, y2) in enumerate(co):
            dx, dy = x1 - x2, y1 - y2
            dist = sqrt(dx * dx + dy * dy)
            matrix[i, j] = dist
    print("Cartesian Matrix:")
    print(matrix)
    print("\n")
    return matrix
#returns Cartesian matrix that has distance of any two points

#get coords from a dictionary based graph
def get_coords(graph):
    coords_list=[]
    for k in range(len(graph)):
        indices = [(k,i) for i, x in enumerate(graph[k]) if x == 1]
        coords_list += indices
        #print(coords_list)
    return coords_list


#create random set of city coords
def create_random(cities, xmax=10, ymax=10):
    co = []
    for j in range(cities):
        x = random.randint(0, xmax)
        y = random.randint(0, ymax)
        co.append((float(x), float(y)))
    print("Inital Graph")

    print(co)
    print("\n")
    return co

#total Counter
def tl(matrix, tour):
    total = 0
    num_cities = len(tour)
    #print(num_cities)
    for i in range(num_cities):
        j = (i + 1) % num_cities
        city_i = tour[i]
        city_j = tour[j]
        total += matrix[city_i, city_j]
    return total

#effeciencey heuristic
def ef(c):
    global cm
    return tl(cm, c)


#instance Class
class I:
    s = 0
    len=0
    seperator = ' '

    def __init__(self, len, c=None, ):

        self.len = len
        self.c = c or self._mc()
        self.s = 0
        self.tour = tour

#randolmy make a a set of alleles
    def _mc(self):
        c = []
        lst = [i for i in range(self.len)]
        for i in range(self.len):
            choice = random.choice(lst)
            lst.remove(choice)
            c.append(choice)
        return c

    def evaluate(self, optimum=None):
        self.s = ef(self.c)

    #recombine
    def cross(self, other):
        l, r = self._pick()
        p1 = I(self.len)
        p2 = I(self.len)
        c1 = [c for c in self.c if c not in other.c[l:r + 1]]
        p1.c = c1[:l] + other.c[l:r + 1] + c1[l:]
        c2 = [c for c in other.c if c not in self.c[l:r + 1]]
        p2.c = c2[:l] + self.c[l:r + 1] + c2[l:]
        return p1, p2

    #mutation on each itereation
    def mutate(self):
        l, r = self._pick()
        temp = self.c[l]
        self.c[l] = self.c[r]
        self.c[r] = temp

    def _pick(self):
        l = random.randint(0, self.len - 2)
        r = random.randint(l, self.len - 1)
        return l, r

    def __repr__(self):
        return '<Tour="%s" Cost=%s>' % (self.seperator.join(map(str, self.c)), self.s)



#environment class
class envi:
    sz = 0
    tour_len =0
    def __init__(self, tour_len, po=None, sz=10, gen=10, rate=0.6, crorate=0.9, muterate=0.1):
        self.sz = sz
        self.tour_len = tour_len
        self.po = self._mpo()
        self.gen = gen
        self.rate = rate
        self.crorate = crorate
        self.muterate = muterate
        for I in self.po:
            I.evaluate()
        self.generation = 0
        self.mins = sys.maxsize
        self.mini = None

    def _mpo(self):
        return [I(self.tour_len) for i in range(0, self.sz)]
    #run the genetic algorithm
    def run(self):
        print("Begin Genetic Algorithm...")
        for i in range(1, self.gen + 1):
            for j in range(0, self.sz):
                self.po[j].evaluate()
                curs = self.po[j].s
                if curs < self.mins:
                    self.mins = curs
                    self.mini = self.po[j]
                    print(self.mini)

            if random.random() < self.crorate:
                children = []
                ni = int(self.rate * self.sz / 2)
                for i in range(0, ni):
                    sel1 = self._rk()
                    sel2 = self._rk()
                    parent1 = self.po[sel1]
                    parent2 = self.po[sel2]
                    c_1, c_2 = parent1.cross(parent2)
                    c_1.evaluate()
                    c_2.evaluate()
                    children.append(c_1)
                    children.append(c_2)
                for i in range(0, ni):
                    sco = 0
                    for k in range(0, self.sz):
                        sco += self.po[k].s

                    r = random.random()
                    a = 0
                    for j in range(0, self.sz):
                        a += (self.po[j].s / sco)
                        if a <= r:
                            self.po[j] = children[i]
                            break
            if random.random() < self.muterate:
                sel = self._select()
                self.po[sel].mutate()
        for i in range(0, self.sz):
            self.po[i].evaluate()
            curs = self.po[i].s
            if curs < self.mins:
                self.mins = curs
                self.mini = self.po[i]
                print(self.mini)
        print("\nBest Outcome:")
        print(self.mini)


    def _select(self):
        sco = 0
        for i in range(0, self.sz):
            sco += self.po[i].s
        r = random.random() * (self.sz - 1)
        a = 0
        sel = 0
        for i in range(0, self.sz):
            a += (1 - self.po[i].s / sco)
            if a <= r:
                sel = i
                break
        return sel
    #fitness threashhold
    def _rk(self, choosebest=0.9):
        #self.po.sort()
        if random.random() < choosebest:
            return random.randint(0, self.sz * self.rate)
        else:
            return random.randint(self.sz * self.rate, self.sz - 1)


def main():
    global cm, tour, num_cities

    maptype = input('Enter File Location for custom map, \nEnter 0 for the Default Map or\n 1 for Random map: ')

    if maptype == '1':
        cities = input("Please enter the number of cities in the map: ")
        if int(cities) > 0:

            maxy = input("Input maximum Y boundary")

            maxX = input ("input maximum X boundary")
            if int(maxy) > 0 and int(maxX) > 0:
                tour = list(range(int(cities)))
                co = create_random(int(cities),int(maxX),int(maxy))
                cm = cartesian_matrix(co)
                ev = envi(len(tour))
                ev.run()
        else:
            print("invalid input...")
            exit(1)

    elif maptype == '0':
        # map of cites where 1 is a city 0 is empty space
        graph = [[1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 1]]

        '''
        solutions: 17.12310562561766
        [2, 4, 3, 0, 1] will be in the same ORDER no matter the starting point
        '''

        print("Inital Graph")
        print(graph)
        print("\n")

        co = get_coords(graph)
        tour = list(range(len(co)))

        cm = cartesian_matrix(co)
        ev = envi(len(tour))
        ev.run()


    elif os.path.isfile(maptype) and os.access(maptype, os.R_OK):

        # set location conditions to a field

        graph = genfromtxt(maptype, delimiter=',')
        co = get_coords(graph)
        tour = list(range(len(co)))
        cm = cartesian_matrix(co)
        ev = envi(len(tour))
        ev.run()

    else:

        print("Not a proper input.. Exiting")

        exit(1)


if __name__ == "__main__":
    main()