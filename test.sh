# running tests on len Seq on CUDA with 8 testing values (not 4)

./CUDAlenSeq 50000 1000 50 10 9 n 0
./CUDAlenSeq 50000 1000 50 10 9 p 0
./CUDAlenSeq 50000 1000 50 10 9 t 0
./CUDAlenSeq 50000 1000 50 10 9 c 0

# running test on threads number with query paral in OMP

./test_nt 100000 1000 50 10 12 1 p1 0

# running test on len seq for thread with all the methods (TODO PRIVATIZATION)

./CPPtest_lenSeq 50000 1000 50 10 9 q 0
./CPPtest_lenSeq 50000 1000 50 10 9 p 0
./CPPtest_lenSeq 50000 1000 50 10 9 p1 0

# running test on len seq on privatization

./CPPtest_lenSeq_P 50000 1000 50 10 9 p1 0