import numpy as np
#This program uses the variational method to optimize the wavefunctions of a hydrogen atom
#The hamiltonian is given by H = -Del^2 * hbar^2/(2m)  - e^2/(4 pi e0 r)
#The basis functions used are eigenfunctions of position (cubes/delta functions), |ri>
#The wavefunction to be found is psi = sum   ci |ri>
#By evaluating and diagonalizing the hamiltonian matrix  <rj|H|ri>,
#the energy eigenfunctions can be found.

#In this basis, U is diagonal, and <ri|U|ri> ~= -e^2/(4 pi e0 ri)
#A large error comes from the approximation of the potential energy, 
#Because it should actually be evaluated by integrating over (xi,xi+d), (yi,yi+d), (zi,zi+d)

#The kinetic energy is evaluated via the Laplacian by the finite difference method
#d^2/dx^2 |ri> = (|ri - d*xhat> -2|ri> + |ri + d*xhat>)/d^2 where d is the grid spacing, and xhat is a unit vector in the x direction

nx = 8 #This program scales as (nx*ny*nz)^3
ny = 8
nz = 8 

d = 0.5 #angstroms Because of the above mentioned error in calculating U, the way the program converges with grid spacing is not immediately obvious
hbarOverM = 7.62 #eV/angstrom^2
eSquaredOverfourPiE0 = 14.39 #eV * angstrom.

def gridPoints(): #Generates the x, y, and z coordinates of the cubic grid. 
#	An even number of points is prefered so that the origin is not included and the singularity is avoided
	xyz = np.zeros((nx*ny*nz,3))
	for i in range(0,nx):
		for j in range(0,ny):
			for k in range(0,nz):
				xyz[i*ny*nz+j*nz+k,0] = d * (i- (nx-1)/2.0);
				xyz[i*ny*nz+j*nz+k,1] = d * (j- (ny-1)/2.0);
				xyz[i*ny*nz+j*nz+k,2] = d * (k- (nz-1)/2.0);
	return xyz			

	
def potentialEnergy(): #Generates a potential energy matrix.
	xyz = gridPoints()
	U = np.zeros((len(xyz),len(xyz)))
	for i in range(0,nx):
		for j in range(0,ny):
			for k in range(0,nz):
				index = i*ny*nz+j*nz+k
				r = (xyz[index,0]**2+xyz[index,1]**2+xyz[index,2]**2) ** 0.5
				U[index,index] = -1 *eSquaredOverfourPiE0/r
	return U


def Laplacian(): #Generates the Laplacian matrix.
	L = np.zeros((nx*ny*nz,nx*ny*nz))
	for i in range(0,nx):
		for j in range(0,ny):
			for k in range(0,nz):
				index = i*ny*nz+j*nz+k
				L[index,index] = -6;				
#When on the boundary of the grid, the approximation of the second derivative changes
#I feel like there must be a better way to do this
				if i>0:
					L[index,index-ny*nz] = 1.0
				else:
					L[index,index] += 1
				if i<nx-1:
					L[index,index+ny*nz] = 1.0
				else:
					L[index,index] += 1
				if j>0:
					L[index,index-nz] = 1.0
				else:
					L[index,index] += 1
				if j<ny-1:
					L[index,index+nz] = 1.0
				else:
					L[index,index] += 1
				if k>0:
					L[index,index-1] = 1.0
				else:
					L[index,index] += 1
				if k<nz-1:
					L[index,index+1] = 1.0
				else:
					L[index,index] += 1
	L = L / d**2
	return L					
	
		
def main(args):
	U = potentialEnergy()
	L = Laplacian()
	T = L * -hbarOverM /2
	H = T+U
#	H = T #Use this expression for the hamiltonian to evaluate a particle-in-a-box instead of the hydrogen atom
	eigE,eigV = np.linalg.eig(H)
	eigE.sort(axis=0) #This separates the eigenvalues from their eigenvectors. Eigenvectors should be rearranged when eigenvalues are sorted.
	print("The lowest energy Eigenvalues:")
	print(eigE[0:20:1]) #print the first 20 eigenvalues
	return 0
    
    
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
