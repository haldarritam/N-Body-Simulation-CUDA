#tables(some types of data structure to store data) we need
#1. table of bodies position
#2. table of force pairs
#3. table of bodies velocity 

#assume all bodies have same mass so F12=F21 and a12=a21

	


def gravitational_accel(pos0,pos1,mass):
	#euclidean distance	
	x,y=pos1[0]-pos0[0],pos1[1]-pos0[1]
	r=sqrt(x**2+y**2)
	
	#[ax,ay]
	return [x*G*mass/(r**3),y*G*mass/(r**3)]

def accel(index,force_pair):
	acceleration=0
	for i in force_pair(num_bodies-1):
		acceleration+=force_pair[i+index*(num_bodies-1)]
	return acceleration
	
#only run once
#def force_pair_index_gen(num_bodies): 
#	force_pair_index=[]
#	for i in range(num_bodies): #0,1,2...N-1
#		for j in range(num_bodies): 
#			force_pair_index.append([])
#			if j>i:
#				force_pair_index[i].append(j+offset)
#			else:
#				force_pair_index[i].append()
#			offset+=num_bodies-1-i
#			
#	return force_pair_index 
	# num_bodies*(num_bodies-1)     #[[0,1,2,...N-2],[0,(N-1),(N-1)+1....(N-1)+(N-3)],[1,(N-1),(N-1)+(N-2),(N-1)+(N-2)+1,...,(N-1)+(N-2)+(N-4)]...[]]
	

for i in range(num_time_step):
	for j in range(num_bodies):
		#force pairs update
		#actual num_force_pairs=num_bodies!/(2!*(num_bodies-2)!)
		#combination problem.... so actually it is able to become O(nC2) instead of O(n^2) (well not too much difference hhhhhh)
		#but coding seems quite complicated 
		p=0
		for i0 in range(num_bodies): 
			for j0 in range(num_bodies):
				if i0!=j0:
					force_pair[p]=gravitational_accel(position_table[i0],position_table[j0],mass)
					p=p+1
				
		# position,acceleration,velocity update
		
		for k in range(num_bodies):
			#v(t+1/2)=v(t-1/2)+a(t)*delta_t
			#x(t+1)=x(t)+v(t+1/2)*delta_t
			#smaller delta_t more accurate
			a=accel(k,force_pair_index,force_pair)*delta_t
			velocity[k][0]=velocity[k][0]+a[0]
			velocity[k][1]=velocity[k][1]+a[1]
			
			pos[k][0]=pos[k][0]+velocity[k][0]*delta_t
			pos[k][1]=pos[k][1]+velocity[k][1]*delta_t
			
			
			
			
			
			