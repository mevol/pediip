# buccaneer makefile

CPPS = $(wildcard ./*.cpp)
OBJS = $(CPPS:%.cpp=%.o)
CC = g++
FLAGS = -O1 -Wall -Wno-sign-compare -fPIC -ftemplate-depth-30
DEBUG = -g
INCDIR = -I${CCP4}/include
LIBDIR = -L${CCP4}/lib
LLIBS = -lclipper-minimol -lclipper-mmdb -lclipper-core -lmmdb2
CFLAGS = ${FLAGS} ${INCDIR} -c
LFLAGS = ${FLAGS} ${LIBDIR} ${LLIBS}

TARGETS = ncacstat

all: ${TARGETS}

ncacstat: ${OBJS}
	${CC} ${LFLAGS} ${OBJS} -o $@

ncacstat.o: ncacstat.cpp
	${CC} ${CFLAGS} $<

clean:
	rm *.o ${TARGETS}
