#Hill_Climb by Daniel Mcdonough 9/18

import random
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


#produce coods from a dictionary of connections
def dict_to_coords(graph):
    coords_list = []
    for key in graph:
        for value in range(len(graph[key])):
            neighbors = graph[key]
            tuple_check= (key,neighbors[value])
            inv=tuple_check[::-1]
            if not inv in coords_list:
                coords_list.append((tuple_check))
    print("Coordinates of Cities:")
    print(coords_list)
    print("\n")
    return coords_list


#returns coods of Cities from a given graph
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
    print("Initial Graph:")
    print(co)
    print("\n")
    return co

#returns the total path cost from a tour
def total(matrix, tour):
    total = 0;
    num_cities = len(tour)
    for i in range(num_cities):
        j = (i + 1) % num_cities
        city_i = tour[i]
        city_j = tour[j]
        total += matrix[city_i, city_j]
    return total

#random tour
def rand_tour(total):
    tour = list(range(total))
    random.shuffle(tour)
    return tour


initial = lambda: rand_tour(len(tour)) #do an initial search
of = lambda tour: total(m, tour) #get the best search of the other opions

#get point pairs
def ap(size, shuffle=random.shuffle):
    r1 = list(range(size))
    r2 = list(range(size))
    if shuffle:
        shuffle(r1)
        shuffle(r2)
    for i in r1:
        for j in r2:
            yield (i, j)

#swap nodes in the tour
def sc(tour):
    for i, j in ap(len(tour)):
        if i < j:
            copy = tour[:]
            copy[i], copy[j] = tour[j], tour[i]
            yield copy

#hill climb  algo
def hillclimb(initial, move_op, of, max_evaluations):
    best = initial()
    b = of(best)
    m = 1
    print("#, Total path Distance, Tour")
    while m < max_evaluations:
        move_made = False
        for next in move_op(best):
            if m >= max_evaluations:
                break
            n = of(next)
            print((m, b, best))
            m += 1
            if n < b:
                best = next
                b = n
                move_made = True
                break
        if not move_made:
            break
    print("\n")
    print(("Hill-Climb Best: ", b, best))




def main():
    global tour, co, m

    maptype = input('Enter File Location for custom map, \nEnter 0 for the Default Map or\n 1 for Random map: ')

    if maptype == '1':

        cities = input("Please enter the number of cities in the map: ")
        if int(cities) > 0:
            co = create_random(int(cities)) #generate city coords
            m = cartesian_matrix(co) #make cartesian matrix
            tour = list(range(int(cities)))
            hillclimb(initial, sc, of, int(cities)*2)
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
        m = cartesian_matrix(co)
        tour = list(range(len(co)))
        hillclimb(initial, sc, of, len(co)*2)
    elif os.path.isfile(maptype) and os.access(maptype, os.R_OK):
        # set location conditions to a field
        graph = genfromtxt(maptype, delimiter=',')
        co = get_coords(graph)
        m = cartesian_matrix(co)
        tour = list(range(len(co)))
        hillclimb(initial, sc, of, len(co) * 2)
    else:
        print("Not a proper input.. Exiting")
        exit(1)


if __name__ == "__main__":
    main()