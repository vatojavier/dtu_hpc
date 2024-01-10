# Makefile
#
TARGET_J  = poisson_j		# Jacobi
TARGET_GS = poisson_gs		# Gauss-Seidel
TARGET_J_OMP = poisson_j_OMP
TARGET_GS_OMP = poisson_gs_OMP

SOURCES	= main.c print.c alloc3d.c
OBJECTS	= print.o alloc3d.o
MAIN_J	= main_j.o
MAIN_GS = main_gs.o
MAIN_J_OMP = main_j_OMP.o
main_GS_OMP = main_gs_OMP.o
OBJS_J	= $(MAIN_J) jacobi.o
OBJS_GS	= $(MAIN_GS) gauss_seidel.o

# options and settings for the GCC compilers
#
CC	= gcc
DEFS	= 
OPT	= -g
IPO	= 
ISA	= 
CHIP	= 
ARCH	= 
PARA	= 
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) $(XOPTS)
LDFLAGS = -lm 

all: $(TARGET_J) $(TARGET_GS) $(TARGET_J_OMP) $(TARGET_GS_OMP)

$(TARGET_J): $(OBJECTS) $(OBJS_J)
	$(CC) -o $@ $(CFLAGS) $(OBJS_J) $(OBJECTS) $(LDFLAGS)

$(TARGET_GS): $(OBJECTS) $(OBJS_GS)
	$(CC) -o $@ $(CFLAGS) $(OBJS_GS) $(OBJECTS) $(LDFLAGS)

$(TARGET_J_OMP): $(OBJECTS) $(OBJS_J_OMP)
	$(CC) -o $@ $(CFLAGS) $(OBJS_J) $(OBJECTS) $(LDFLAGS)

$(TARGET_GS_OMP): $(OBJECTS) $(OBJS_GS_OMP)
	$(CC) -o $@ $(CFLAGS) $(OBJS_GS) $(OBJECTS) $(LDFLAGS)

$(MAIN_J):
	$(CC) -o $@ -D_JACOBI $(CFLAGS) -c main.c 

$(MAIN_GS):
	$(CC) -o $@ -D_GAUSS_SEIDEL $(CFLAGS) -c main.c 

$(MAIN_J_OMP):
	$(CC) -o $@ -D_JACOBI $(CFLAGS) -fopenmp -c main.c 

$(MAIN_GS_OMP):
	$(CC) -o $@ -D_GAUSS_SEIDEL $(CFLAGS) -fopenmp -c main.c 

clean:
	@/bin/rm -f core *.o *~

realclean: clean
	@/bin/rm -f $(TARGET_J)  $(TARGET_GS)

# DO NOT DELETE

main_j.o: main.c print.h jacobi.h 
main_gs.o: main.c print.h gauss_seidel.h
print.o: print.h
