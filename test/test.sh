# running tests on len Seq on CUDA with 8 testing values (not 4)

#./CUDAlenSeq 50000 1000 50 10 9 n 0
#./CUDAlenSeq 50000 1000 50 10 9 p 0
#./CUDAlenSeq 50000 1000 50 10 9 t 0
#./CUDAlenSeq 50000 1000 50 10 9 c 0


# running tests on len Query on CUDA with 7 testing values

./CUDAlenQuery 50000 100 50 10 7 n 0
./CUDAlenQuery 50000 100 50 10 7 p 0
./CUDAlenQuery 50000 100 50 10 7 t 0
./CUDAlenQuery 50000 100 50 10 7 c 0

# running tests on num Queries on CUDA with 6 testing values

./CUDAnumQueries 50000 1000 50 10 6 n 0
./CUDAnumQueries 50000 1000 50 10 6 p 0
./CUDAnumQueries 50000 1000 50 10 6 t 0
./CUDAnumQueries 50000 1000 50 10 6 c 0

# running test on threads number with query paral in OMP

#./CPPtest_nt 100000 1000 50 10 12 1 p1 0

# running test on len seq for thread with all the methods (TODO PRIVATIZATION)

#./CPPtest_lenSeq 50000 1000 50 3 9 12 s 0
#./CPPtest_lenSeq 50000 1000 50 10 9 12 p 0
#./CPPtest_lenSeq 50000 1000 50 10 9 12 p1 0

# running test on len seq on privatization

#./CPPtest_lenSeq_P 50000 1000 50 10 9 12 p1 0

