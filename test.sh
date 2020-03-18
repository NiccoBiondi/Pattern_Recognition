# running tests on len Seq on CUDA with 8 testing values (not 4)

./CUDAlenSeq 100000 1000 50 10 8 n 0
./CUDAlenSeq 100000 1000 50 10 8 p 0
./CUDAlenSeq 100000 1000 50 10 8 t 0
./CUDAlenSeq 100000 1000 50 10 8 c 0

# running test on threads number with query paral in OMP

./test_nt 100000 1000 50 10 12 1 q 0
