OBJ := main
INC := ./include
LIB := ./lib
CC := nvcc
SRC := $(OBJ).cu
OSRC := $(OBJ).o
OBJFLAGS := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -ccbin g++ -I $(INC) 
CFLAGS := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -ccbin g++ -L $(LIB) -lsfml-graphics -lsfml-window -lsfml-system 

all : main.o
	$(CC) $(CFLAGS) $(OSRC) -o $(OBJ)
	rm $(OSRC)

main.o:
	$(CC) $(OBJFLAGS) -c $(SRC)

clean :
	rm $(OBJ)
