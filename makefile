CXX = g++
CFLAGS = -Wall -O3 -Wextra -std=c++17
DEP := distribuzione.o


distribuzione.exe: $(DEP)
	$(CXX) $(CFLAGS) $^ -o $@

%.o : %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -rf *.exe *.o
	echo PULIZIA COMPLETATA