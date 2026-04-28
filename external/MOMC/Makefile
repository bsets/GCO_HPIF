CC ?= gcc
CFLAGS ?= -O3 -DMOMC
SRC := src/MOMC2016_1800sec_timeout_aware.c
BIN := bin/MoMC

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p bin
	$(CC) $(CFLAGS) $(SRC) -o $(BIN)
	chmod +x $(BIN)

clean:
	rm -f $(BIN)
