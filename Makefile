OBJ := cpu
INC := ./include
LIB := ./lib
CC := g++
SRC := $(OBJ).c
OSRC := $(OBJ).o
OBJFLAGS := -I $(INC) 
CFLAGS := -L $(LIB) -lsfml-graphics -lsfml-window -lsfml-system -pthread 

all : main.o
	$(CC) $(CFLAGS) $(OSRC) -o $(OBJ)
	#g++ -I ./include $(LIB)  -Wall  -g -o cpu.o cpu.c -lm -lsfml-graphics -lsfml-window -lsfml-system
	rm $(OSRC)

main.o:
	$(CC) $(OBJFLAGS) -c $(SRC)
	
	


clean :
	rm $(OBJ)
