import numpy as np
import pygame as pg

## PYGAME SETUP
pg.init()
screen = pg.display.set_mode((1920, 1080))
clock = pg.time.Clock()
pg.display.set_caption("SpringTopes V1.0")
running = True
dt = 0

## CONSTANTS
# F = edge spring force
F = 0.05
# FC = coplanar spring force
FC = 0.15
# RF = regularity force
RF = 0.05
# ZS = zoom speed
ZS = 1.01
# CD = cooldown (in seconds)
CD = 0.2
if F > 0.6:
    F = 0.6
# D is only used to set up the tetrahedron
D = 50
# L is the edge length
L = 120
FONT = pg.font.Font("freesansbold.ttf", 16)
framerate = 60

## GLOBAL VARIABLES
# set up base tetrahedron:
allPoints = [pg.Vector3(D,D,D), pg.Vector3(-D,D,-D), pg.Vector3(D,-D,-D), pg.Vector3(-D,-D,D)]
allFaces = [[0,1,2],[0,2,3],[0,3,1],[1,3,2]]
stress = [0,0,0,0]
colors = [(255,0,0), (0,255,0), (0,0,255), (127,126,0)]
move = [pg.Vector3(0,0,0)]
for i in range(len(allPoints)-1):
    move.append(pg.Vector3(0,0,0))
selectedTri = 0
selectedPoint = 0
GAS = 0
orders = [0]

zoom = 1
cam = pg.Vector3(1,0,0)
saveM = [0,0]
saveR = 0
saveT = 0
rot = 0
tilt = 0
opCool = 0
newPoints = 0
saveColor = (0,0,0)
rf = True
edgeStress = False
faceStress = False
gui = 0

# set up the base tetrahedron, all faces should always be clockwise
def resetToTet():
    global allPoints
    global allFaces
    global colors
    global move
    global selectedTri
    global selectedPoint
    global GAS
    global orders
    global stress

    allPoints = [pg.Vector3(D,D,D), pg.Vector3(-D,D,-D), pg.Vector3(D,-D,-D), pg.Vector3(-D,-D,D)]
    allFaces = [[0,1,2],[0,2,3],[0,3,1],[1,3,2]]
    stress = [0,0,0,0]
    colors = [(255,0,0), (0,255,0), (0,0,255), (127,126,0)]
    move = [pg.Vector3(0,0,0)]
    for i in range(len(allPoints)-1):
        move.append(pg.Vector3(0,0,0))
    selectedTri = 0
    selectedPoint = 0
    GAS = 0
    orders = [0]

# transforms a point to its coordinates on the screen
def transform(point):
    pointR = pg.Vector3(0,point.y,0)
    pointR.x = np.arctan2(point.z,point.x)
    pointR.z = np.sqrt(np.square(point.x) + np.square(point.z))

    output = pg.Vector2(0,0)
    output.x = pointR.z * np.sin(rot + pointR.x)
    output.y = np.cos(rot + pointR.x)*np.sin(tilt) * pointR.z + pointR.y * np.cos(tilt)

    output *= zoom

    output.x += screen.get_width() / 2
    output.y += screen.get_height() / 2

    return output

# finds the normal vector of a set of points
def findNormal(P):
    # find the shape's normal vector through averaging cross products (optimize this later if you can figure it out)
    Xa = P
    Xa = [i for i in Xa if i != 0]
    Xb = [Xa[-1]] + Xa[:-1]
    Xa = [Xa_i - Xb_i for Xa_i, Xb_i in zip(Xa, Xb)]
    Xb = [Xa[-1]] + Xa[:-1]
    Xa = [np.cross(Xa_i, Xb_i) for Xa_i, Xb_i in zip(Xa, Xb)]
    Xb = sum(Xa)
    if np.linalg.norm(Xb) == 0:
        return pg.Vector3(0,0,0.01)
    else:
        return -Xb / np.linalg.norm(Xb)

# prepares a new polygon
def createPoly(RGB,P):
    normal = findNormal(P)
    difference = 1 - np.arccos(np.dot(normal,cam)) / np.pi * 2
    if np.isnan(difference):
        difference = 1

    # decide where onscreen (x,y) to render points
    shape = P
    shape = [i for i in shape if i != 0]
    shape = [transform(i) for i in shape]

    # find centroid
    avg = sum(P, pg.Vector3(0,0,0)) / len(P)

    if difference < 0:
        orderPoly(tuple([np.abs(difference)*x for x in RGB]), shape, avg)

# orders the polygon to the order list
def orderPoly(RGB, xypos, centroid):
    global orders
    layer = np.linalg.norm(centroid - cam * 5000)
    if np.isnan(layer) == True:
        layer = 0
    layer = int(layer*10000)
    if orders == [0]:
        orders = [[layer, [RGB], xypos]]
    else:
        orders.append([layer, [RGB], xypos])

# gives True if M is within the triangle formed by a, b, and c
def isWithinTriangle(M,a,b,c):
    if (((b[0] - a[0])*(M[1] - a[1]) > (b[1] - a[1])*(M[0] - a[0])) & ((c[0] - b[0])*(M[1] - b[1]) > (c[1] - b[1])*(M[0] - b[0])) & ((a[0] - c[0])*(M[1] - c[1]) > (a[1] - c[1])*(M[0] - c[0]))):
        return True
    else:
        return False

# extension of triangle, limited to pentagons:
def isWithinPoly(M,P):
    global allPoints
    n = 0
    result = False
    for i in range(len(P)-2):
        result = result or isWithinTriangle(M,transform(allPoints[P[0]]),transform(allPoints[P[i+1]]),transform(allPoints[P[i+2]]))
    return result

# adds new random colors to the list
def newColor(L = 1):
    for i in range(L):
        newColors = [np.random.randint(0,256)]
        newColors.insert(np.random.randint(0,2), 255 - newColors[0])
        newColors.insert(np.random.randint(0,3), 0)

        colors.append(tuple(newColors))
if len(colors) < len(allFaces):
    newColor(len(allFaces) - len(colors))

# orders faces by their depths
def sortFaceDepth():
    global orders
    global allFaces
    global colors
    # order every face by depth to prepare for tetrahedral augmentations
    for i in range(len(allFaces)):
        avg = sum([allPoints[p] for p in allFaces[i]], pg.Vector3(0,0,0)) / len(allFaces[i])
        layer = np.linalg.norm(avg - cam * 5000)
        if np.isnan(layer) == True:
            layer = 0
        layer = int(layer*10000)

        if orders == [0]:
            orders = [[layer, i]]
        else:
            orders.append([layer, i])
    
    # similar ordering algorithm is used later for rendering
    orders = [y[1] for y in sorted(orders, key= lambda x: x[0])]
    newFaces = allFaces[:]
    newColors = colors[:]

    for i in range(len(allFaces)):
        newFaces[i] = allFaces[orders[i]]
        newColors[i] = colors[orders[i]]
    
    allFaces = newFaces[:]
    colors = newColors[:]
    orders = [0]

# orders points by their depths
def sortPointDepth():
    global orders
    global newPoints

    for i in range(len(allPoints)):
        layer = np.linalg.norm(allPoints[i] - cam * 5000)
        if np.isnan(layer) == True:
            layer = 0
        layer = int(layer*10000)

        if orders == [0]:
            orders = [[layer, i]]
        else:
            orders.append([layer, i])
        
    orders = [y[1] for y in sorted(orders, key= lambda x: x[0])]
    newPoints = allPoints[:]

    # the only purpose of newPoints being here is to ensure that the nearest point to the camera if multiple are clicked at once is the one calculated
    for i in range(len(allPoints)):
        newPoints[i] = allPoints[orders[i]]

while running:
    ## RESET (#X to print, #J + #L to reset, #ESC to leave)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    screen.fill("black")

    if pg.key.get_pressed()[pg.K_ESCAPE]:
        running = 0
    
    if pg.key.get_pressed()[pg.K_x] & (opCool <= 0):
        # copy components to clipboard
        truncPoints = [x[:] for x in allPoints[:]]
        for x in truncPoints:
            for i in range(len(x)):
                x[i] = int(x[i])
        save = "[" + str(truncPoints[:]) + ", " + str(allFaces[:]) + ", " + str(colors[:]) + ", " + str(int(GAS*100)) + ", " + str(rf) + "]"
        pg.scrap.put_text(save)
        opCool = CD * framerate

    if pg.key.get_pressed()[pg.K_j] & pg.key.get_pressed()[pg.K_l]:
        resetToTet()

    ## CAMERA (#I zoom in, #O zoom out, #CLICK pan camera)
    if pg.key.get_pressed()[pg.K_o]:
        zoom *= ZS
    if pg.key.get_pressed()[pg.K_i]:
        zoom *= 1 / ZS

    if pg.mouse.get_just_pressed()[0] == True:
        saveM = pg.mouse.get_pos()
        saveR = rot
        saveT = tilt

    if pg.mouse.get_pressed()[0] == True:
        rot = saveR + (pg.mouse.get_pos()[0] - saveM[0]) / 200
        tilt = saveT + (pg.mouse.get_pos()[1] - saveM[1]) / 200
        tilt = np.minimum(tilt,np.pi/2)
        tilt = np.maximum(tilt,-np.pi/2)

    # prevent rot from becoming an unnessacarily large value
    if rot >= np.pi*2:
        rot = rot - np.pi*2
    if rot < 0:
        rot = rot + np.pi*2

    # calculate camera vector:
    cam = pg.Vector3(np.cos(rot) * np.cos(tilt), -np.sin(tilt), -np.sin(rot) * np.cos(tilt))

    #R toggle regularity force
    if pg.key.get_pressed()[pg.K_r] & (opCool <= 0):
        if rf:
            rf = False
        else:
            rf = True
        opCool = CD * framerate

    #A pyramidial augmentations
    if pg.key.get_pressed()[pg.K_a] & (opCool <= 0):
        # make sure you are calculating the relevant (visible) faces
        sortFaceDepth()

        # execute augmentation
        for x in allFaces:
            if (selectedTri != x) & isWithinPoly(pg.Vector2(pg.mouse.get_pos()),x):
                normal = -findNormal((allPoints[x[0]],allPoints[x[1]],allPoints[x[2]]))
                ## place point a bit off of the centroid
                centroid = [allPoints[x[i]] for i in range(len(x))]
                centroid = sum(centroid, pg.Vector3(0,0,0)) / len(x)
                newPos = centroid + normal * 0.8 * L
                allPoints.append(pg.Vector3(newPos))
                move.append(pg.Vector3(0,0,0))

                # remove the old face and create a pyramid with the new:
                oldFace = x
                colors.pop(allFaces.index(x))
                allFaces.remove(x)

                for i in range(len(oldFace)):
                    allFaces.append([len(allPoints) - 1,oldFace[i-1],oldFace[i]])

                newColor(len(oldFace))

                # no more operations for a half second
                opCool = framerate * CD
                break

    #D tetrahedral diminishing (the opposite of A)
    if pg.key.get_pressed()[pg.K_d] & (opCool <= 0):
        # make sure you get the closest point to the camera
        sortPointDepth()

        for i in range(len(newPoints)):
            # check the local vertex group
            localFaces = [x for x in allFaces if orders[i] in x]
            vertexType = [len(x) for x in localFaces]

            if (np.linalg.norm(pg.Vector2(pg.mouse.get_pos()) - transform(newPoints[i])) < 15) & (set(vertexType) == {3}):
                # remove the three diminished faces:
                for x in localFaces:
                    colors.pop(allFaces.index(x))
                    allFaces.remove(x)
                
                # create a new face based on existing data:
                newFace = localFaces[:]
                for p in range(len(newFace)):
                    newFace[p] = newFace[p][newFace[p].index(orders[i]):] + newFace[p][:newFace[p].index(orders[i])]
                    newFace[p] = newFace[p][1:3]
                
                # order newFace's "dominoes" and extract the new face:
                sortFace = [newFace[0]]
                newFace.pop(0)

                # count is here to prevent it from making an infinite loop
                count = 0
                while (len(newFace) > 0) & (count < 20):
                    count += 1
                    for x in newFace:
                        if x[0] == sortFace[-1][1]:
                            sortFace.append(x)
                            newFace.remove(x)
                            break
                newFace = [sortFace[n][0] for n in range(len(sortFace))]

                allFaces.append(newFace)
                newColor()

                # remove the diminished point from main lists:
                for j in range(len(allFaces)):
                    for l in range(len(allFaces[j])):
                        if allFaces[j][l] >= orders[i]:
                            allFaces[j][l] -= 1
                
                move.pop(orders[i])
                allPoints.pop(orders[i])

                # no more operations for a half second
                opCool = framerate * CD
                break
        orders = [0]

    #E disphenoid swap
    if pg.key.get_pressed()[pg.K_e] & (opCool <= 0):
        if (selectedTri == 0):
            sortFaceDepth()
            for x in allFaces:
                if (len(x) == 3) & isWithinTriangle(pg.Vector2(pg.mouse.get_pos()),transform(allPoints[x[0]]),transform(allPoints[x[1]]),transform(allPoints[x[2]])):
                    selectedTri = x
                    saveColor = colors[allFaces.index(x)]
                    colors[allFaces.index(x)] = (255,255,255)
                    # don't do any more operations until the second face is selected
                    opCool = CD * framerate
                    break
        else:
            failed = 1
            sortFaceDepth()
            for x in allFaces:
                if ((len(list(set(x).intersection(selectedTri))) == 2) & (len(x) == 3)) & isWithinTriangle(pg.Vector2(pg.mouse.get_pos()),transform(allPoints[x[0]]),transform(allPoints[x[1]]),transform(allPoints[x[2]])):
                    failed = 0
                    # lazy way of making the tris end with their shared components
                    firstTri = selectedTri[:]
                    secondTri = x
                    for i in range(2):
                        if (firstTri[0] == list(set(secondTri).intersection(firstTri))[0]) or (firstTri[0] == list(set(secondTri).intersection(firstTri))[1]):
                            firstTri.append(firstTri[0])
                            firstTri.pop(0)
                        if (secondTri[0] == list(set(secondTri).intersection(firstTri))[0]) or (secondTri[0] == list(set(secondTri).intersection(firstTri))[1]):
                            secondTri.append(secondTri[0])
                            secondTri.pop(0)
                    
                    # change some points to flip the triangles
                    firstTri[2] = secondTri[0]
                    secondTri[2] = firstTri[0]

                    allFaces[allFaces.index(x)] = firstTri
                    colors[allFaces.index(selectedTri)] = saveColor
                    allFaces[allFaces.index(selectedTri)] = secondTri

                    selectedTri = 0
                    opCool = CD * framerate
                    break
            if failed == 1:
                colors[allFaces.index(selectedTri)] = saveColor
                selectedTri = 0
                opCool = CD * framerate

    #F triangular fold (previous but instead removes the edge)
    if pg.key.get_pressed()[pg.K_f] & (opCool <= 0):
        if (selectedTri == 0):
            sortFaceDepth()
            for x in allFaces:
                if (len(x) == 3) & isWithinTriangle(pg.Vector2(pg.mouse.get_pos()),transform(allPoints[x[0]]),transform(allPoints[x[1]]),transform(allPoints[x[2]])):
                    selectedTri = x
                    saveColor = colors[allFaces.index(x)]
                    colors[allFaces.index(x)] = (255,255,255)
                    # don't do any more operations until the second face is selected
                    opCool = CD * framerate
                    break
        else:
            failed = 1
            sortFaceDepth()
            for x in allFaces:
                if ((len(list(set(x).intersection(selectedTri))) == 2) & (len(x) == 3)) & isWithinTriangle(pg.Vector2(pg.mouse.get_pos()),transform(allPoints[x[0]]),transform(allPoints[x[1]]),transform(allPoints[x[2]])):
                    failed = 0
                    
                    firstTri = selectedTri[:]
                    secondTri = x[:]
                    shared = list(set(secondTri).intersection(firstTri))

                    # find the non-shared points:
                    firstTri.remove(shared[0])
                    firstTri.remove(shared[1])
                    secondTri.remove(shared[0])
                    secondTri.remove(shared[1])
                    
                    # merge the non-shared points:
                    for i in range(len(allFaces)):
                        for j in range(len(allFaces[i])):
                            if allFaces[i][j] == secondTri[0]:
                                allFaces[i][j] = firstTri[0]
                            if allFaces[i][j] > secondTri[0]:
                                allFaces[i][j] -= 1

                    del colors[allFaces.index(selectedTri)]
                    allFaces.remove(selectedTri)
                    del colors[allFaces.index(x)]
                    allFaces.remove(x)

                    allPoints[firstTri[0]] = (allPoints[firstTri[0]] + allPoints[secondTri[0]]) / 2
                    move.pop(secondTri[0])
                    allPoints.pop(secondTri[0])

                    selectedTri = 0
                    opCool = CD * framerate
                    break
            if failed == 1:
                colors[allFaces.index(selectedTri)] = saveColor
                selectedTri = 0
                opCool = CD * framerate

    #Q adding/removing edges
    if pg.key.get_pressed()[pg.K_q] & (opCool <= 0):
        if selectedPoint == 0:
            sortPointDepth()
            # find what point you are clicking
            for x in newPoints:
                if np.linalg.norm(pg.Vector2(pg.mouse.get_pos()) - transform(x)) < 15:
                    selectedPoint = x
                    opCool = CD * framerate
                    break
        else:
            failed = 1
            sortPointDepth()
            # find secondary point
            for x in allPoints:
                if np.linalg.norm(pg.Vector2(pg.mouse.get_pos()) - transform(x)) < 15:
                    pA = allPoints.index(x)
                    pB = allPoints.index(selectedPoint)
                    testEdges = [y for y in allFaces if all(z in (y + [y[0]]) for z in [pA,pB]) or all(z in (y + [y[0]]) for z in [pB,pA])]
                    testFaces = [y for y in allFaces if (pA in y) and (pB in y)]
                    
                    # either make or break an edge:
                    if len(testEdges) == 2: # MAKE
                        failed = 0
                        
                        # convoluted way to sort into a new face
                        facesHere = testEdges[:]

                        if all(z in (facesHere[0] + [facesHere[0][0]]) for z in [pB,pA]):
                            facesHere.reverse()
                        
                        # cycle pA to the front of the lists so they can be merged:
                        while facesHere[0][0] != pA:
                            facesHere[0].append(facesHere[0][0])
                            facesHere[0].pop(0)

                        while facesHere[1][0] != pA:
                            facesHere[1].append(facesHere[1][0])
                            facesHere[1].pop(0)
                        
                        if facesHere[0][-1] == pB:
                            facesHere.reverse()
                        
                        facesHere[0].pop(0)
                        facesHere[1].pop()
                        facesHere = facesHere[0] + facesHere[1]

                        # create the new face and remove its components:
                        newColor()
                        allFaces.append(facesHere)
                        del colors[allFaces.index(testEdges[0])]
                        allFaces.remove(testEdges[0])
                        del colors[allFaces.index(testEdges[1])]
                        allFaces.remove(testEdges[1])
                        
                        # prevent the new face from creating a "lone" point
                        loneTest = allFaces[-1][:]
                        if len(loneTest) != len(set(loneTest)):
                            while loneTest[0] != loneTest[2]:
                                loneTest.append(loneTest[0])
                                loneTest.pop(0)
                            
                            del allPoints[loneTest[1]]
                            for j in range(len(allFaces)):
                                for l in range(len(allFaces[j])):
                                    if allFaces[j][l] >= loneTest[1]:
                                        allFaces[j][l] -= 1
                            del loneTest[2]
                            del loneTest[1]

                        selectedPoint = 0
                        opCool = CD * framerate
                        break

                    elif len(testFaces) == 1: # BREAK
                        failed = 0
                        cuttingFace = testFaces[0][:]

                        while cuttingFace[0] != pA:
                            cuttingFace.append(cuttingFace[0])
                            cuttingFace.pop(0)
                        
                        faceA = cuttingFace[:(cuttingFace.index(pB) + 1)]
                        faceB = [cuttingFace[0]] + cuttingFace[(cuttingFace.index(pB)):]

                        newColor(2)
                        allFaces.append(faceA)
                        allFaces.append(faceB)
                        del colors[allFaces.index(testFaces[0])]
                        allFaces.remove(testFaces[0])

                        selectedPoint = 0
                        opCool = CD * framerate
                        break

            if failed == 1:
                selectedPoint = 0
                opCool = CD * framerate
            
        orders = [0]

    #1/2/3 setting stress modes
    if pg.key.get_pressed()[pg.K_1]:
        edgeStress = False
        faceStress = False
    elif pg.key.get_pressed()[pg.K_2]:
        edgeStress = True
        faceStress = False
    elif pg.key.get_pressed()[pg.K_3]:
        edgeStress = False
        faceStress = True

    ## PHYSICS (#W add gas, #S remove gas)
    stress = [0 for i in allFaces]

    # allow some springiness from last frame
    for i in range(len(move)):
        move[i] *= 0.5

    # gas controller
    if pg.key.get_pressed()[pg.K_w] == True:
        for i in range(len(move)):
            GAS += 0.02

    if pg.key.get_pressed()[pg.K_s] == True:
        for i in range(len(move)):
            GAS -= 0.02
            if GAS < 0:
                GAS = 0

    # calculate edge springs
    for i in range(len(allFaces)):
        for l in range(len(allFaces[i])):
            diff = allPoints[allFaces[i][l]] - allPoints[allFaces[i][l-1]]
            if np.linalg.norm(diff) != 0:
                target = [pg.Vector3(L*diff/np.linalg.norm(diff) - diff)]
            else:
                target = pg.Vector3(0,0,0.01)
            if round(sum(target,pg.Vector3(0,0,0))) != [0,0,0]:
                move[allFaces[i][l]] += target[0] * F
                move[allFaces[i][l-1]] -= target[0] * F
                if edgeStress:
                    stress[i] += np.abs(np.linalg.norm(target[0] * F))
                    print(stress)

    # force coplanarity, !apply gas!, and force regularity if on
    for i in range(len(allFaces)):
        # gas force applied to normals (like a balloon)
        face = list(allPoints[allFaces[i][l]] for l in range(len(allFaces[i])))
        normal = findNormal(face)
        for x in allFaces[i]:
            move[x] -= normal * GAS
        
        # does not apply to triangles:
        if len(allFaces[i]) > 3:
            centroid = sum(face, pg.Vector3(0,0,0)) / len(face)
            for l in range(len(allFaces[i])):
                # coplanarity force:
                target = normal * np.dot((centroid - allPoints[allFaces[i][l]]),normal)
                move[allFaces[i][l]] += target * FC
                if faceStress:
                    stress[i] += np.abs(np.linalg.norm(target * FC))

                # regular faced force:
                if rf == True:
                    r = L * 0.5 / np.sin(np.pi / len(allFaces[i]))
                    dist = allPoints[allFaces[i][l]] - centroid
                    if np.linalg.norm(dist) != 0:
                        target = (dist / np.linalg.norm(dist)) * (np.linalg.norm(dist) - r)
                    else:
                        target = pg.Vector3(0,0,0.01)
                    move[allFaces[i][l]] -= target * RF
                    if edgeStress:
                        stress[i] += np.abs(np.linalg.norm(target * RF))

    # EMERGENCY: automatically ends program if something gets too laggy
    if move[0][0] > 9999:
        running = 0
    
    # apply movements
    for i in range(len(allPoints)):
        allPoints[i] += move[i]

    # center shape again for good measure
    for x in allPoints:
        x -= sum(allPoints, pg.Vector3(0,0,0)) / len(allPoints)

    ## RENDERING

    # set up for rendering
    for i in range(len(allFaces)):
        if edgeStress or faceStress:
            stress[i] *= 100
            if stress[i] > 255:
                stress[i] = 255
            stressIndicator = (stress[i], 255 - stress[i], 0)
            
            createPoly(stressIndicator,[allPoints[x] for x in allFaces[i]])
        else:
            createPoly(colors[i],[allPoints[x] for x in allFaces[i]])

    # sort the orders by depth and render one by one
    orders = sorted(orders, key= lambda x: -x[0])
    for i in range(len(orders)):
        pg.draw.polygon(screen, tuple(orders[i][1]), orders[i][2])
    orders = [0]

    if selectedPoint != 0:
        pg.draw.circle(screen, (255,255,255), pg.Vector2(transform(selectedPoint)), 5)

    ## GUI
    # rf vs non-rf indicator (square vs rhombus)
    sw = screen.get_width()
    if rf:
        pg.draw.polygon(screen,(0,255,0),[(sw-10,10), (sw-10,25), (sw-25,25), (sw-25,10)])
    else:
        pg.draw.polygon(screen,(255,0,0),[(sw-10,10), (sw-20,20), (sw-35,20), (sw-25,10)])

    # info button
    pg.draw.circle(screen,(10,120,220),(30,30),15)
    pg.draw.circle(screen,(255,255,255),(22,30),2)
    pg.draw.circle(screen,(255,255,255),(30,30),2)
    pg.draw.circle(screen,(255,255,255),(38,30),2)

    # show the user tips if the button is pressed
    if (np.linalg.norm(pg.Vector2(pg.mouse.get_pos()) - pg.Vector2((30,30))) <= 15) & pg.mouse.get_just_pressed()[0]:
        if gui != 1:
            gui = 1
        else:
            gui = 0
    
    if pg.key.get_pressed()[pg.K_y] & (opCool <= 0):
        if gui != 2:
            gui = 2
        else:
            gui = 0
        opCool = CD * framerate

    # info GUI
    if gui == 1:
        text = "INFO:" \
        "\nThis is intended to be a polyhedron constructor / viewer." \
        "\nTo save your work, you must copy the polyhedron to your clipboard (X), " \
        "\nthen paste it into the import menu (Y) of any future instances." \
        "\n\nIf the program closes, it will be due to pressing ESC, facing" \
        "\na fatal error, or having a polyhedron explode infinitely." \
        "\n\nVIEWPORT:" \
        "\nClick and drag to pan in 3D space, and use I/O to zoom in and out." \
        "\nPress 1/2/3 to change viewing modes:" \
        "\n     1: Normal-colored" \
        "\n     2: Edge-stress" \
        "\n     3: Coplanarity-stress" \
        "\nR: Toggle faces attempting to approach regular angles." \
        "\nW/S: Change \"gas\" pressure (normal force on every face)" \
        "\nJ+L: Reset to tetrahedron" \
        "\n\nOPERATIONS:" \
        "\nA/D: Augment / diminish a pyramid" \
        "\n     - Select one face / pyramid point" \
        "\nQ: Add or remove an edge to merge or break faces" \
        "\n     - Select two points on the same face" \
        "\nE: Perform a disphenoid swap" \
        "\n     - Select two adjacent triangles" \
        "\nF: Fold & glue two triangles together" \
        "\n     - Ditto" \
        "\n(\"select\" as in press a key while the mouse is nearest the desired part)"
        text = FONT.render(text, True, (200,200,200))
        screen.blit(text, (60,60))

    # import GUI
    if gui == 2:
        centX = screen.get_width() / 2
        centY = screen.get_height() / 2
        menu = pg.Rect(0,0,300,110)
        menu.center = (centX,centY)
        pg.draw.rect(screen,(50,50,50),menu)
        
        text = "Press [0] to import from clipboard." \
        "\nIf nothing happens, you do not" \
        "\nhave the right thing copied." \
        "\n\nPress Y to cancel."

        text = FONT.render(text, True, (200,200,200))
        screen.blit(text, (centX - 140,centY - 43))

        if pg.key.get_just_pressed()[pg.K_0]:
            try:
                data = eval(pg.scrap.get_text())
                if isinstance(data, list):
                    resetToTet()
                    allPoints = [pg.Vector3(x) for x in data[0]]
                    move = [pg.Vector3(0,0,0) for i in allPoints]
                    allFaces = data[1]
                    colors = data[2]
                    GAS = data[3] / 100
                    rf = data[4]
                    gui = 0
            except:
                gui = gui

    # restricts the framerate- dt is delta time, just in case it is needed.
    opCool -= 1
    pg.display.flip()
    dt = clock.tick(framerate) / 1000

pg.quit()