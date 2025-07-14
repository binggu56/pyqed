#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 08:41:22 2025

@author: bingg

https://github.com/xenoscopic/clebsch-gordan/blob/master/ClebschGordan.py
"""

from math import sqrt
import math
from time import sleep
from PIL import Image, ImageDraw, ImageFont

#"Constants" for rendering
DefaultCellHeight = 10
DefaultCellWidth = 40
DefaultThinLineWidth = 1
DefaultThickLineWidth = 1

#Some utility methods needed for quantum mechanics...
def cgFactorial(n):
	if n < 0:
		return -10000000000000000000000.0  #Technically the factorial of a negative number is -infinity,
	if math.floor(n) != n:
		raise ValueError("n must be exact integer")
	if n+1 == n:  # catch a value like 1e300
		raise OverflowError("n too large")
	result = 1
	factor = 2
	while factor <= n:
		result *= factor
		factor += 1
	return result

def frange(start, end=None, inc=None):
	"A range function, that does accept float increments..."

	if end == None:
		end = start + 0.0
		start = 0.0

	if inc == None:
		inc = 1.0

	L = []
	while 1:
		next = start + len(L) * inc
		if inc > 0 and next >= end:
			break
		elif inc < 0 and next <= end:
			break
		L.append(next)
	return L

#end utility methods

def calculateCoefficient(j1, j2, m1, m2, j, m):

    delta = (m == (m1 + m2))

    srNum = (2*j + 1)*cgFactorial(j + j1 - j2)*cgFactorial(j - j1 + j2)*cgFactorial(j1 + j2 - j)*cgFactorial(j + m)*cgFactorial(j - m)
    srDen = cgFactorial(j1 + j2 + j + 1)*cgFactorial(j1 - m1)*cgFactorial(j1 + m1)*cgFactorial(j2 - m2)*cgFactorial(j2 + m2)
    sumList = [((((-1)**(k + j2 + m2))*cgFactorial(j2 + j + m1 - k)*cgFactorial(j1 - m1 + k))/(cgFactorial(k)*cgFactorial(j - j1 + j2 - k)*cgFactorial(j + m - k)*cgFactorial(k + j1 - j2 - m))) for k in range(0, int(math.floor(j + m + 1)))]
    sumList = [s for s in sumList if (abs(s) > .00000000000000001)] #Filter out those values where the numerator should have been infinite and killed this off

    return delta*sqrt(srNum/srDen)*sum(sumList)

def getCondonShortleyForm(coeff):
	sign = 1
	if coeff < 0:
		sign = -1

	sqr = coeff**2
	denom = 1
	testNum = sqr

	while True:
		testNum = sqr * denom
		if abs(round(testNum) - testNum) < .00001: #Denominator is pretty close to an integer when multiplied by test numerator
			break
		denom += 1

	return (sign, int(round(testNum)), denom)

class tableWedge:
	def __init__(self,  j1,  j2,  jValues, bigM):
		self.contents = {}
		self.isNegativeM = bigM < 0
		jValues.sort() #They should be sorted already, but...
		jValues.reverse()
		self.jValues = jValues
		self.mVal = bigM
		self.m1m2Values = {}
		m1 = j1 if j1 < bigM + j2 else bigM + j2 #Get the starting maximum m1 on the left
		m2 = bigM - m1
		#Loop over rows
		for row in range(0,  len(jValues)):
			#Loop over columns (there will be as many columns as rows)
			self.m1m2Values[row] = (m1, m2)
			for col in range(0,  len(jValues)):
				self.contents[(row,  col)] = getCondonShortleyForm(calculateCoefficient(float(j1),  float(j2),  float(m1),  float(m2),  float(jValues[col]), float(self.mVal))) #These are being converted to ints for some reason, I'm not sure why, but we can just convert them here
			m1 -= 1
			m2 = bigM - m1

	def render(self, draw, font, divot):
		sideLen = len(self.jValues)

		#Render surrounding box
		coord1 = (divot[0], divot[1] - DefaultCellHeight*2 - 2)
		coord2 = (coord1[0] + DefaultCellWidth*sideLen - 4, coord1[1])
		coord3 = (coord2[0], coord2[1] + DefaultCellHeight*(2 + sideLen) + 4)
		coord4 = (coord3[0] - DefaultCellWidth*(2 + sideLen) - 2, coord3[1])
		coord5 = (coord4[0], coord4[1] - DefaultCellHeight*sideLen - 2)
		boxCoords = [divot, coord1, coord2, coord3, coord4, coord5, divot]
		draw.line(boxCoords, width = DefaultThickLineWidth)

		#Render inside lines
		draw.line([divot, (coord2[0], divot[1])], width = DefaultThinLineWidth) #Horizontal
		draw.line([divot, (divot[0], coord3[1])], width = DefaultThinLineWidth) #Vertical

		#Render all the text
		for row in range(0, sideLen):
			#Render the m1 and m2
			m1, m2 = self.m1m2Values[row]
			m1Negative, m2Negative = m1 < 0, m2 < 0
			m1 = getCondonShortleyForm(sqrt(abs(m1)))
			m2 = getCondonShortleyForm(sqrt(abs(m2)))
			#These long strings format the fraction appropriately, taking care of sign, denominator
			m1Text = "%s%s/%s" % ("-" if m1Negative else "+", m1[1], m1[2]) if m1[2] != 1 else "%s%s" % ("-" if m1Negative else "+", m1[1])
			m2Text = "%s%s/%s" % ("-" if m2Negative else "+", m2[1], m2[2]) if m2[2] != 1 else "%s%s" % ("-" if m2Negative else "+", m2[1])
			m1X, m1Y = coord5[0] + 1, coord5[1] + 1 + row*DefaultCellHeight
			m2X, m2Y = coord5[0] + 1 + DefaultCellWidth, coord5[1] + 1 + row*DefaultCellHeight
			draw.text((m1X, m1Y), m1Text, font=font)
			draw.text((m2X, m2Y), m2Text, font=font)
			for col in range(0, sideLen):
				#Render J and M (only if this is the first row in the column)
				if row == 0:
					jTot = self.jValues[col]
					bigM = self.mVal
					bigMNegative = bigM < 0
					jTot, bigM = getCondonShortleyForm(sqrt(jTot)), getCondonShortleyForm(sqrt(abs(bigM)))
					jTotText = "%s/%s" % (jTot[1], jTot[2]) if jTot[2] != 1 else "%s" % (jTot[1])
					bigMText = "%s%s/%s" % ("-" if bigMNegative else "+", bigM[1], bigM[2]) if bigM[2] != 1 else "%s%s" % ("-" if bigMNegative else "+", bigM[1])
					jTotCoords = (coord1[0] + 1 + col*DefaultCellWidth, coord1[1] + 1)
					bigMCoords = (coord1[0] + 1 + col*DefaultCellWidth, coord1[1] + 1 + DefaultCellHeight)
					draw.text(jTotCoords, jTotText, font=font)
					draw.text(bigMCoords, bigMText, font=font)

				#Render the coefficient
				coeff = self.contents[(row, col)]
				text = "%s%s/%s" % ("-" if coeff[0] < 0 else "", coeff[1], coeff[2]) if coeff[2] != 1 else "%s%s" % ("-" if coeff[0] < 0 else "", coeff[1])
				textX, textY = (divot[0] + 1) + col*DefaultCellWidth, (divot[1] + 1) + row*DefaultCellHeight
				draw.text((textX, textY), text, font=font)
		#Return new divot
		return coord3

	def getWidthFromDivot(self):
		return len(self.jValues)*DefaultCellWidth - 4

	def getHeightFromDivot(self):
		return len(self.jValues)*DefaultCellHeight + 2

def buildTable(j1, j2):

    jTot = j1 + j2
    
	#Loop over wedges
    wedges = []
    numWedges = int(2*jTot+ 1)
	
    physicallyRealizableJMin = abs(j1 - j2)
    
    for wedgeIndex in range(0,  numWedges):
		#These formulas here are a pain in the ass and can't really be written out nicely.  Suffice it to say, they're correct
        wedgeBasedJMin = jTot - wedgeIndex if wedgeIndex <= jTot else wedgeIndex - jTot
        actualJMin = physicallyRealizableJMin if wedgeBasedJMin < physicallyRealizableJMin else wedgeBasedJMin
		#print(physicallyRealizableJMin, wedgeBasedJMin, actualJMin)
		
        jValues = frange(jTot,  actualJMin - 1,  -1)
        
        bigM = jTot - wedgeIndex
        wedges.append(tableWedge(j1,  j2,  jValues, bigM))
    
    return wedges

def renderTable(j1, j2, path, wedges):
    jTot = j1 + j2

	#Calculate the necessary image size
    divotPosition = (2*DefaultCellWidth + 6, 2*DefaultCellHeight + 2)
    imgWidth = sum([wedge.getWidthFromDivot() for wedge in wedges]) + divotPosition[0] + 1
    imgHeight = sum([wedge.getHeightFromDivot() for wedge in wedges]) + divotPosition[1] + 1

	#P = palette, (width, height), white background
	#See http://effbot.org/zone/creating-palette-images.htm
    img = Image.new('P',(imgWidth,imgHeight), 255)
    draw = ImageDraw.Draw(img)
# 	jFont = ImageFont.truetype("DejaVu.ttf", 15)
    jFont = ImageFont.load_default()
# 	cFont = ImageFont.truetype("DejaVu.ttf", 10)

    cFont = ImageFont.load_default()


	#Render the big J stuff
    j1 = getCondonShortleyForm(sqrt(j1)) #Safe to pass in sqrts because j1, j1 shouldn't be negative, although we need TODO double check this
    j2 = getCondonShortleyForm(sqrt(j2))
    j1Str = "%i/%i" % (j1[1], j1[2]) if j1[2] != 1 else "%i" % j1[1]
    j2Str = "%i/%i" % (j2[1], j2[2]) if j2[2] != 1 else "%i" % j2[1]
    draw.text((0,0), "%s X %s" % (j1Str, j2Str), font=jFont)

	#Positions specified as (x, y), y runs downward
    for wedge in wedges:
        divotPosition = wedge.render(draw, cFont, divotPosition)

	#Save file
    # try:
    #     img.save(path)
    # except(KeyError):
    #     print("Couldn't save due to invalid file path.")

    # img.save(fp='test.png')
    img.show()

if __name__ == "__main__":

# 	path = raw_input("Image save path: ")

	#Get j1 and j2
    validateJ = lambda j: math.floor(j/.5) == j/.5

# 	j1 = raw_input("j1 (in decimal form): ")
    j1 = 1/2
# 	try:
# 		j1 = float(j1)
# 		if not validateJ(j1):
# 			j1 = .5
# 			print("j1 must be a multiple of 1/2, idiot.  Going with j1 = 1/2.")
# 	except:
# 		j1 = .5
# 		print("Invalid value of j1 input.  Going with j1 = 1/2.")
    j2 = 1/2
# 	j2 = raw_input("j2 (in decimal form): ")
# 	try:
# 		j2 = float(j2)
# 		if not validateJ(j2):
# 			j2 = .5
# 			print("j2 must be a multiple of 1/2, idiot.  Going with j2 = 1/2.")
# 	except:
# 		j2 = .5
# 		print("Invalid value of j2 input.  Going with j2 = 1/2.")

	#Allow user to override cell height/width
# 	cellWidth = raw_input("Cell Width [Default]: ")
    cellWidth = None
    if cellWidth:
        try:
            DefaultCellWidth = int(cellWidth)
        except:
            print("Invalid width, going with default value of %i" % DefaultCellWidth)

# 	cellHeight = raw_input("Cell Height [Default]: ")
    cellHeight = 2

# 	if cellHeight:
# 		try:
# 			DefaultCellHeight = int(cellHeight)
# 		except:
# 			print("Invalid height, going with default value of %i" % DefaultCellHeight)

	#DO IT!
    # m1 = 1/2
    # m2 = 1/2
    # j = 1
    # m =0
    # c = calculateCoefficient(j1, j2, m1, m2, j, m)
    # print(c)
    # path = 'test'
    wedges = buildTable(j1, j2)

    # print(wedges[0].jValues, wedges[0].mVal)
    path = 'test.png'
    renderTable(j1, j2, path, wedges)