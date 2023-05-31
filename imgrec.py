import numpy
import math
import cv2
import scipy
import scipy.spatial
import time
import datetime
import tkinter as tk
import RPi.GPIO as GPIO
import json
import os

def guassian(x, s):
    e = - (x**2) / (2 * (s**2))
    rf = math.exp(e)
    lf = 1 / math.sqrt(2 * math.pi * (s**2))
    return lf * rf

gaussian_size = 127
gaussian_sigma = 1.7
guassian_kernel = numpy.zeros((gaussian_size, gaussian_size))

for y in range(gaussian_size):
    for x in range(gaussian_size):
        r = (gaussian_size - 1)/2
        gy = guassian(y - r, gaussian_sigma)
        gx = guassian(x - r, gaussian_sigma)
        guassian_kernel[y][x] = gy * gx

############################

cellCount = 5

configFilename = "config.json"

if os.path.isfile("./" + configFilename):
    print("loading config")
    with open(configFilename, "r") as f:
        config = json.load(f)

    cellAmounts = config["cellAmounts"]
    doseTime = config["doseTime"]
    lastDoseTimestamp = config["lastDoseTimestamp"]
else:

    cellAmounts = [0] * cellCount
    doseTime = 0
    lastDoseTimestamp = 0

state = "idle"

buzzerPin = 11
buzzerDutyCycle = 0.005
buzzerStates = {
    "dose": (0.005, 5),
    "refill": (0.005, 1),
    "success": (0.001, 1500),
    "idle": (0.0, 1000)
}

GPIO.setmode(GPIO.BOARD)
GPIO.setup(buzzerPin, GPIO.OUT)
GPIO.output(buzzerPin, GPIO.LOW)

buzzerPWM = GPIO.PWM(buzzerPin, buzzerStates[state][1])
buzzerPWM.start( buzzerStates[state][0] )

############################

window = tk.Tk()
window.title("Medicine Organiser")

label = tk.Label(window, text="Time:")
label.grid(row=1, column=0, padx=(10,10), pady=0)

timePartEntries = [None]*2
timePartLabels = ["HH", "MM"]
timePartInits = [ doseTime // 60, doseTime % 60 ]

for x in range(2):
    label = tk.Label(window, text=timePartLabels[x])
    label.grid(row=0, column=x+1, padx=(10,10), pady=(10,0))

    timePartEntries[x] = tk.Entry(window, width=5)
    timePartEntries[x].insert(0, str( timePartInits[x] ))
    timePartEntries[x].grid(row=1, column=x+1, padx=(10,10), pady=0)

label = tk.Label(window, text="Amounts:")
label.grid(row=3, column=0, padx=(10,10), pady=0)

countEntries = [None]*cellCount

for x in range(cellCount):

    label = tk.Label(window, text=f"M{x}")
    label.grid(row=2, column=x+1, padx=(10,10), pady=(10,0))

    countEntries[x] = tk.Entry(window, width=5)
    countEntries[x].insert(0, str( cellAmounts[x] ))
    countEntries[x].grid(row=3, column=x+1, padx=(10,10), pady=0)

def resetLastDoseTimestamp():
    print("Reset last dose timestamp")
    global lastDoseTimestamp
    lastDoseTimestamp = 0

def submit():
    for x in range(cellCount):
        cellAmounts[x] = int(countEntries[x].get())

    hour = int(timePartEntries[0].get())
    minute = int(timePartEntries[1].get())
    doseTime = hour*60 + minute

    print(doseTime)

    config = {
        "cellAmounts": cellAmounts,
        "doseTime": doseTime,
        "lastDoseTimestamp": lastDoseTimestamp
    }

    with open(configFilename, "w") as f:
        json.dump(config, f)

    window.destroy()

confirmButton = tk.Button(window, text="Confirm", command=submit)
confirmButton.grid(row=4, columnspan=math.ceil(cellCount/2), padx=(10,10), pady=10)

resetButton = tk.Button(window, text="Reset clock", command=resetLastDoseTimestamp)
resetButton.grid(row=4, column=math.ceil(cellCount/2), columnspan=math.floor(cellCount/2), padx=(10,10), pady=10)

window.mainloop()

#exit()

############################

medCount = numpy.sum(cellAmounts)

############################

############################

vid = cv2.VideoCapture(0)

if vid is None or not vid.isOpened():
    print("Camera error")
    exit()
else:
    print("Camera working")

vid.set(cv2.CAP_PROP_FOCUS, 1)

ret, frame = vid.read()
vidH = frame.shape[0]
vidW = frame.shape[1]

cropX = 0
cropY = -50
cropW = math.floor(vidW / 2.25)
cropH = math.floor(vidH / 2.25)

mismatchFrames = 0
matchFrames = 0

emptyFrames = 0
nonEmptyFrames = 0

def rectSort(rect):
    return rect[0]

while(True):
    time.sleep(0.1)

    _, frame = vid.read()

    # crop frame
    frame = frame[
        cropY + (vidH - cropH)//2 : cropY + cropH + (vidH - cropH)//2, 
        cropX + (vidW - cropW)//2 : cropX + cropW + (vidW - cropW)//2
    ]
    frame = cv2.resize(frame, (vidW, vidH))

    # save original frame
    source = frame
    
    # filter frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.filter2D(frame, 1, guassian_kernel)
    frame = cv2.convertScaleAbs(frame)
    frame = cv2.Canny(frame, 30, 100)
    
    # detect shapes
    contours , _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [] if contours is None else list(contours)

    # reverse filtering
    frame = source

    # filter small or duplicate contours
    filteredContours = []
    
    i = 0
    for contour in contours:
        if cv2.arcLength(contour, True) > 30:
            filteredContours.append(contour)
        i += 1

    contours = filteredContours

    xData = numpy.zeros( (1, len(contours)) )
    yData = numpy.zeros( (1, len(contours)) )
    for index, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        xData[0][index] = approx.ravel()[0]
        yData[0][index] = approx.ravel()[1]
    kd_tree : scipy.spatial.KDTree = scipy.spatial.KDTree(numpy.c_[xData.ravel(), yData.ravel()])
    
    filteredContours = []

    removed = set()
    for index, contour in enumerate(contours):
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        d, j = kd_tree.query([[0,0], [x, y]], k=2)
        
        if index not in removed and j[1][1] not in removed:
            filteredContours.append(contour)
            if abs(d[0][0] - d[0][1]) < 10.0:
                removed.add( j[1][1] )

    contours = filteredContours

    ##################

    # determine which contours are cells and which are meds

    cells = []
    meds = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
        a = approx.reshape( approx.shape[0]*approx.shape[1], 2 )

        left = numpy.amin(a[:, 0])
        top = numpy.amin(a[:, 1])
        right = numpy.amax(a[:, 0])
        bottom = numpy.amax(a[:, 1])
        
        cv2.putText(frame, f"{len(approx)}", (left, top), color=(255,0,255), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0)
        cv2.drawContours(frame, [approx], contourIdx=0, color=(0,255,0), thickness=1)

        if 4 <= len(approx) <= 6:
            cells.append((left, top, right, bottom))
        else:

            cX = left + (right - left)//2
            cY = top + (bottom - top)//2

            sub = source[ top:bottom, left:right ]
            if sub.shape[0] != 0 and sub.shape[1] != 0:
                avg = sub[ sub.shape[0]//2, sub.shape[1]//2 ]
                avg = (int(avg[0]), int(avg[1]), int(avg[2]))
            else:
                avg = (0,0,0)

            meds.append((cX, cY, avg))

    cells.sort(key=rectSort)

    #################################

    for index, med in enumerate(meds):
        cX, cY, avg = med
        cv2.circle(frame, (cX, cY), 15, avg, -1)
        cv2.putText(frame, f"{index}", (cX, cY), color=(255,0,0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75)

    cv2.putText(frame, f"{len(contours)}", (vidW//2,vidH//2), color=(255,0,255), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0)


    allFilled = True
    allEmpty = True
    if len(cells) == cellCount:
        for cellIndex, cell in enumerate(cells):
            left, top, right, bottom = cell

            contains = []
            for medIndex, med in enumerate(meds):
                cX, cY, _ = med

                flag = (cY > top) and (cY < bottom) and (cX > left) and (cX < right)
                if flag:
                    contains.append(medIndex)

            filled = len(contains) == cellAmounts[cellIndex]
            allFilled = allFilled and filled
            allEmpty = allEmpty and (len(contains) == 0)

            text = ""
            if len(contains) > cellAmounts[cellIndex]:
                text += "/TOO MANY"
            elif len(contains) < cellAmounts[cellIndex]:
                text += "/TOO FEW"



            cv2.rectangle(frame, (left, top), (right, bottom), (255 if filled else 0, 0, 255), 1)
            cv2.putText(frame, f"{text}", (left, top), color=(255,0,0), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75)

    if not allFilled:
        matchFrames = 0
        mismatchFrames += 1
    else:
        mismatchFrames = 0
        matchFrames += 1
        
    if allEmpty:
        nonEmptyFrames = 0
        emptyFrames += 1
    else:
        emptyFrames = 0
        nonEmptyFrames += 1
        
    

    if state == "idle":
        if mismatchFrames > 5:
            state = "refill"
            
    if state == "refill":
        if matchFrames > 5:
            state = "idle"
            
    if state == "success":
        state = "idle"
            
    if state == "dose":
        if emptyFrames > 5:
            state = "success"
            
            now = datetime.datetime.now()
            sinceSunrise = now.hour*60*60 + now.minute*60 + now.second
            lastDoseTimestamp = (time.time() - sinceSunrise) + doseTime*60
            
            with open(configFilename, "r") as f:
                config = json.load(f)
            config["lastDoseTimestamp"] = lastDoseTimestamp
            with open(configFilename, "w") as f:
                json.dump(config, f)
            
            
            
    if state != "dose":
        nowTS = time.time()
        if nowTS - lastDoseTimestamp > 60*60*24:
            now = datetime.datetime.now()
            if now.hour*60 + now.minute >= doseTime:
                state = "dose"
                print("Going to dose state")

        
    buzzerPWM.ChangeDutyCycle(buzzerStates[state][0])
    buzzerPWM.ChangeFrequency(buzzerStates[state][1])

    ######################################
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

vid.release()
cv2.destroyAllWindows()

buzzerPWM.stop()
GPIO.cleanup()

