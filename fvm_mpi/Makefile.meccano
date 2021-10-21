APP = calcium_sparks
DOUBLE = 1
DEBUG = 0

# Compiler options
CC = mpicc
VECTORIZE = -ftree-vectorize -ftree-vectorizer-verbose=0 -msse -msse2 -msse3 
CFLAGS = -O3 -Wall -Wno-comment -Wno-unknown-pragmas -g -march=native -mtune=native -fPIC -ffast-math -std=gnu89 -I/usr/include/hdf5/openmpi #$(VECTORIZE)
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/
LIBS = -lhdf5 -lm

# Source files
HEADERS = utils.h options.h geometry.h species.h types.h boundaryfluxes.h mtwist.h basefluxinfo.h
SOURCES = utils.c options.c geometry.c species.c boundaryfluxes.c mtwist.c $(APP).c 

# Source directories
SRC_DIR = src/
OBJ_DIR = build/
BIN_DIR = bin/

# Get object files from sources by swaping .c by .o
OBJS = $(SOURCES:.c=.o)

# To every file add a prefix with an appropriate directory
CSRC=$(addprefix $(SRC_DIR),$(SOURCES))
CHEAD=$(addprefix $(SRC_DIR),$(HEADERS))
COBJS=$(addprefix $(OBJ_DIR),$(OBJS))
CAPP=$(addprefix $(BIN_DIR),$(APP))

#Compilation rules
$(CAPP): $(COBJS)
	$(CC) $(CFLAGS) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} -o $(CAPP)$(APP_SUFFIX) $(COBJS) $(LIB_DIRS) $(LIBS)

$(OBJ_DIR)%.o : $(SRC_DIR)%.c $(CHEAD)
	$(CC) $(CFLAGS) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} $(INC) -c $< -o $@

all: $(COBJS) $(CAPP)

$(COBJS): | $(OBJ_DIR)
$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(CAPP): | $(BIN_DIR)
$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm $(COBJS) $(CAPP)
