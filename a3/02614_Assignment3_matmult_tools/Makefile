TARGET= libmatmult.so
OBJS	= matmult_c.o

CC	= nvc
CXX	= nvc++

OPT	= -g -fast -Msafeptr -Minfo -mp=gpu -gpu=pinned -gpu=lineinfo -gpu=cc90 -cuda -mp=noautopar -cudalib=cublas
PIC   = -fpic -shared
ISA	=
PARA	=
INC   =
LIBS	=

CXXFLAGS= $(OPT) $(PIC) $(INC) $(ISA) $(PARA) $(XOPT)

all: $(TARGET)

$(TARGET): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	@/bin/rm -f $(TARGET) $(OBJS)
