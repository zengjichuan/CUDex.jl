CFLAGS=-O3 -use_fast_math

libcudex.so: basic.o
	nvcc $(CFLAGS) --shared --compiler-options -fPIC $^ -o $@

%.o: %.cu
	nvcc -c $(CFLAGS) --compiler-options -fPIC $< -o $@

%.cu: %gen.jl %.jl
	julia $< > $@

clean:
	-rm *.o *.cu libcudex.so

# .PRECIOUS: *.cu
