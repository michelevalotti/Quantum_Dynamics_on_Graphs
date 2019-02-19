import cv2

print('press x to exit')

while(True):
	var = 'q'
	for i in range(100):
		# 2dwalkvidPot or 2dwalkvidMarked or 2dwalkvidMarkedPlot or 2dwalkvid
		# or 2dwalkvid2Part or 2dwalkvid2PartTorus or 2dwalkvidDensM or 2dwalkvidLoss
		# or 2dwalkvidRepulsion or 2dwalkvidRepulsionLoss or 2dwalkvidVarEdges
		frame = cv2.imread('2dwalkvidMarked'+str(i)+'.jpg') 
		cv2.imshow('image', frame)
		k = cv2.waitKey(100) & 0xFF
		if k == ord('x'): #esc key closes display window
			var = 'x'
			break
	k = cv2.waitKey(1000) & 0xFF
	if var == 'x':
		break