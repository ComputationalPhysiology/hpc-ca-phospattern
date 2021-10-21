#make -j OMP=1 STREAM=1
APP = calcium_sparks
DOUBLE = 1
DEBUG = 0

ifndef $(CC)
CC = mpiicc
CXX = icpc
else
CC = mpicc -cc=gcc
CXX = g++
endif

ifndef $(ARCH)
ARCH = 4
endif

ifndef $(MANUAL)
MANUAL=2
endif

ifeq ($(STREAM),1)
STREAMFLAG = -DSTREAM
endif

ifeq ($(SWAP),1)
SWAPFLAG = -DSWAP
endif

ifeq ($(OMP),1)
OMPFLAG = -fopenmp
endif

ifeq ($(MANUAL),1)
MANUALFLAG = -DMANUAL
endif

ifeq ($(ALPHA),1)
ALPHAFLAG = -DALPHA
endif

ifeq ($(MANUAL),2)
MANUALFLAG = -DMANUAL -DMANUAL2
endif

ifeq ($(NOVEC),1)
NOVECFLAG = -no-vec -no-simd
endif

ifeq ($(SIMD),1)
SIMDFLAG = -DSIMDFLAG
endif

ifeq ($(GATHERONLY),1)
GATHERFLAG = -DGATHERONLY
endif


ifeq ($(ARCH),1)
ARCHFLAG = -msse4.1 #ICC and GCC
endif

ifeq ($(ARCH),2)
ARCHFLAG = -mavx
endif

ifeq ($(ARCH),3)
ifeq ($(CC), mpiicc)
ARCHFLAG = -march=core-avx2 -fma #ICC
else
ARCHFLAG = -mavx2 -mfma
endif
endif

ifeq ($(ARCH),4)
ifeq ($(CC), mpiicc)
#ARCHFLAG = -xCOMMON-AVX512  #ICC
ARCHFLAG = -xMIC-AVX512  #ICC
else
ARCHFLAG = -mavx512f -mfma
endif
endif

# Source directories
SRC_DIR = src/
OBJ_DIR = build/
BIN_DIR = bin/

# Compiler options
#CC = mpiicc
#CXX = icpc
#CC = mpicc
#CXX = g++

CFLAGS = -Ofast -Wall -Wno-comment -Wno-unknown-pragmas -Wformat=0 $(ARCHFLAG) -std=gnu11 -fgnu89-inline -I/usr/include/hdf5/openmpi
CXXFLAGS = -Ofast -Wall -Wno-comment -Wno-unknown-pragmas -Wformat=0 -fno-exceptions -I/usr/include/hdf5/openmpi -I$(SRC_DIR)vectorclass
LIB_DIRS = -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/

#CFLAGS = -Ofast -Wall -Wno-comment -Wno-unknown-pragmas -Wformat=0 $(ARCHFLAG) -std=gnu11 -fgnu89-inline -I/usr/lib/x86_64-linux-gnu/hdf5/mpich/include 
#CXXFLAGS = -Ofast -Wall -Wno-comment -Wno-unknown-pragmas -Wformat=0 -I$(SRC_DIR)vectorclass
#LIB_DIRS = -L/usr/lib/x86_64-linux-gnu/hdf5/mpich/
LIBS = -lhdf5 -lm

# Source files
HEADERS = utils.h options.h geometry.h species.h types.h boundaryfluxes.h mtwist.h basefluxinfo.h kernel.h
SOURCES = utils.c options.c geometry.c species.c boundaryfluxes.c mtwist.c $(APP).c 

# Get object files from sources by swaping .c by .o
OBJS = $(SOURCES:.c=.o)
OBJS += kernel_SIMD.o
#OBJS += kernel_AVX512.o
#OBJS += kernel_AVX2.o
#OBJS += kernel_SSE4.o

# To every file add a prefix with an appropriate directory
CSRC=$(addprefix $(SRC_DIR),$(SOURCES))
CHEAD=$(addprefix $(SRC_DIR),$(HEADERS))
COBJS=$(addprefix $(OBJ_DIR),$(OBJS))
CAPP=$(addprefix $(BIN_DIR),$(APP))

#Compilation rules
$(CAPP): $(COBJS)
	$(CC) $(CFLAGS) $(STREAMFLAG) $(SWAPFLAG) $(MANUALFLAG) $(ALPHAFLAG) $(GATHERFLAG) $(OMPFLAG) $(NOVECFLAG) $(SIMDFLAG) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} -o $(CAPP)$(APP_SUFFIX) $(COBJS) $(LIB_DIRS) $(LIBS)

$(OBJ_DIR)%.o : $(SRC_DIR)%.c $(CHEAD)
	$(CC) $(CFLAGS) $(STREAMFLAG) $(SWAPFLAG) $(MANUALFLAG) $(ALPHAFLAG) $(GATHERFLAG) $(OMPFLAG) $(NOVECFLAG) $(SIMDFLAG) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} $(INC) -c $< -o $@

$(OBJ_DIR)kernel_SIMD.o : $(SRC_DIR)kernel_generic.cpp $(CHEAD)
	$(CXX) $(CXXFLAGS) $(ARCHFLAG) $(STREAMFLAG) $(SWAPFLAG) $(MANUALFLAG) $(ALPHAFLAG) $(GATHERFLAG) $(OMPFLAG) $(NOVECFLAG) $(SIMDFLAG) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} $(INC) -c $< -o $@

# $(OBJ_DIR)kernel_AVX512.o : $(SRC_DIR)kernel_generic.cpp $(CHEAD)
# 	$(CXX) $(CXXFLAGS) -xMIC-AVX512 $(STREAMFLAG) $(MANUALFLAG) $(ALPHAFLAG) $(GATHERFLAG) $(OMPFLAG) $(NOVECFLAG) $(SIMDFLAG) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} $(INC) -c $< -o $@

# $(OBJ_DIR)kernel_AVX2.o : $(SRC_DIR)kernel_generic.cpp $(CHEAD)
# 	$(CXX) $(CXXFLAGS) -march=core-avx2 $(STREAMFLAG) $(MANUALFLAG) $(ALPHAFLAG) $(GATHERFLAG) $(OMPFLAG) $(NOVECFLAG) $(SIMDFLAG) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} $(INC) -c $< -o $@

# $(OBJ_DIR)kernel_SSE4.o : $(SRC_DIR)kernel_generic.cpp $(CHEAD)
# 	$(CXX) $(CXXFLAGS) -msse4.1 $(STREAMFLAG) $(MANUALFLAG) $(ALPHAFLAG) $(GATHERFLAG) $(OMPFLAG) $(NOVECFLAG) $(SIMDFLAG) -DDOUBLE=${DOUBLE} -DDEBUG=${DEBUG} $(INC) -c $< -o $@

all: $(COBJS) $(CAPP)

$(COBJS): | $(OBJ_DIR)
$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(CAPP): | $(BIN_DIR)
$(BIN_DIR):
	mkdir $(BIN_DIR)

clean:
	rm $(COBJS) $(CAPP)
