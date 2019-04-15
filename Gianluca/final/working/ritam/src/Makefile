OBJ := gpu2_streams
INC := ./include
LIB := ./lib
CC := nvcc
SRC := ./src/*
OSRC := $(OBJ).o
OBJDIR := ./obj
OBJFLAGS := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -ccbin clang++ -I $(INC) 
CFLAGS := -arch=sm_52 -ccbin clang++ -L $(LIB) -lsfml-graphics -lsfml-window -lsfml-system 
CFLAGS += -m64 -Xptxas -dlcm=cg  -Xcompiler="-pthread -Wall -O3"

all : main.o
	$(CC) $(CFLAGS) $(OBJDIR)/* -o $(OBJ)

main.o:
	$(CC) $(OBJFLAGS) -c $(SRC)
	mv *.o $(OBJDIR)

clean :
	rm $(OBJ)
