TARGET= poisson
OBJS	= main.o \
        jacobi.o \
        init.o \
        input.o \
        print.o \
        alloc3d.o \

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=pinned -gpu=cc90 -gpu=lineinfo -cuda -mp=noautopar
ISA	=
PARA	=
INC   =
LIBS	=

CFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)
CXXFLAGS = $(OPT) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)
clean:
	@/bin/rm -f *.o $(TARGET)
