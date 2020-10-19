# Import library
import math
import matplotlib.pyplot as plt
import numpy as np

def  plt_scara(model, L, corner):
	
	n=100
	#x=np.array([ np.linspace(-0.5,0.5, n),  np.linspace(0.5,0.5, n), np.linspace(0.5,-0.5,n), np.linspace(-0.5,-0.5, n),]).reshape(4*n)
	#y=np.array([np.linspace(-0.5,-0.5,n), np.linspace(-0.5,0.5, n), np.linspace(0.5,0.5, n),  np.linspace(0.5, -0.5, n)]).reshape(4*n)
	
	#corner=[-0.3,-0.0]
	x=np.array([ np.linspace(0,L, n),  np.linspace(L,L, n), np.linspace(L,0,n), np.linspace(0,0, n),]).reshape(4*n)
	x=x+corner[0]
	y=np.array([np.linspace(0, 0,n), np.linspace(0,L, n), np.linspace(L,L, n),  np.linspace(L, 0, n)]).reshape(4*n)
	y=y+corner[1]
	EE=np.array([x,y]).T
	q_pred=(model.predict(EE))
	# Input Arm length
	l1 = 0.5
	l2 = 0.5


	# Define Angle variable
	theta_1= q_pred[:,0]
	theta_2= q_pred[:,1]
	# Input Position of (x0,y0)
	x0 = 0
	y0 = 0


	# A constant for getting movie frame.
	k=1
	positions_x = []
	positions_y = []
	plt.figure(1)
	plt.xlim([-1.5, 1.5])
	plt.ylim([-1.5, 1.5])
	for i in range(theta_1.shape[0]):
		plt.clf()
        # Calculate coordinates (x1, y1)
		x1= l1* math.cos(theta_1[i])
		y1= l1* math.sin(theta_1[i])

        # Calculate (x2,y2)
		x2= x1 + l2*math.cos(theta_1[i] + theta_2[i])
		y2= y1 + l2*math.sin(theta_1[i] + theta_2[i])
		positions_x.append(x2)
		positions_y.append(y2)
        
        # Plot end effector path
		plt.scatter(positions_x, positions_y, color='C1')
    
        # Define plot file name for generate animation
		#filename = str(k) + '.png'
		k=k+1
        
        # Plot axis limit
		plt.axis('equal')
		plt.xlim([-1.5, 1.5])
		plt.ylim([-1.5, 1.5])
        
        # Save plot figure
        #plt.savefig(filename)
		plt.plot([x[0], x[n-1], x[2*n-1], x[3*n-1], x[4*n-1]], [y[0], y[n-1], y[2*n-1], y[3*n-1], y[4*n-1]], '--')
        
        # Plot of robotics arm
		plt.plot([x0,x1], [y0,y1],'r', linewidth=10)
		plt.plot([x1,x2], [y1,y2],'b', linewidth=10)
		plt.show()
		plt.pause(0.02)
	return 

