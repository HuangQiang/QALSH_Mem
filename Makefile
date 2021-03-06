SRCS=random.cc pri_queue.cc util.cc kd_node.cc kd_tree.cc \
	qalsh.cc qalsh_plus.cc ann.cc main.cc
OBJS=${SRCS:.cc=.o}

CXX=g++ -std=c++11
CPPFLAGS=-w -O3

.PHONY: clean

all: ${OBJS}
	${CXX} ${CPPFLAGS} -o qalsh ${OBJS}

random.o: random.h

pri_queue.o: pri_queue.h

util.o: util.h

kd_node.o: kd_node.h

kd_tree.o: kd_tree.h

qalsh.o: qalsh.h

qalsh_plus.o: qalsh_plus.h

ann.o: ann.h

main.o:

clean:
	-rm ${OBJS}
