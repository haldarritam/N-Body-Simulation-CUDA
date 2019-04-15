OBJ := everything
INC := ./include
LIB := ./lib
CC := nvcc
SRC := $(OBJ).cu
OSRC := $(OBJ).o
OBJFLAGS := -arch=sm_52 -ccbin clang++ -I $(INC) 
CFLAGS := -arch=sm_52 -ccbin clang++ -L $(LIB) -lsfml-graphics -lsfml-window -lsfml-system 

all : main.o
	$(CC) $(CFLAGS) $(OSRC) -o $(OBJ)
	rm $(OSRC)

main.o:
	$(CC) $(OBJFLAGS) -c $(SRC)

clean :
	rm $(OBJ)
