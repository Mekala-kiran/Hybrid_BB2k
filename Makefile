NVCC     = nvcc
ARCH     = -arch=sm_75
CFLAGS   = -std=c++17 -O3 -Xcompiler="-fopenmp -pthread"
LIBS     = -ltbb
TARGET   = BFC_VP
SRC      = BFC_VP.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)

