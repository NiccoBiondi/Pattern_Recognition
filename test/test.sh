# running tests on len Seq on CUDA with 8 testing values (not 4)

./CUDAlenSeq 50000 1000 10 10 9 n 0
./CUDAlenSeq 50000 1000 10 10 9 p 0
./CUDAlenSeq 50000 1000 10 10 9 t 0
./CUDAlenSeq 50000 1000 10 10 9 c 0

./CUDAnumQueries 100000 1000 10 10 6 n 0
./CUDAnumQueries 100000 1000 10 10 6 p 0
./CUDAnumQueries 100000 1000 10 10 6 t 0

./CUDAlenQuery 100000 100 10 10 7 n 0
./CUDAlenQuery 100000 100 10 10 7 p 0
./CUDAlenQuery 100000 100 10 10 7 t 0
./CUDAlenQuery 100000 100 10 10 7 c 0



# running tests on len Query on CUDA with 7 testing values

./CUDAthb32 100000 1000 10 10 1 n 0
./CUDAthb64 100000 1000 10 10 1 n 0
./CUDAthb128 100000 1000 10 10 1 n 0
./CUDAthb256 100000 1000 10 10 1 n 0
./CUDAthb512 100000 1000 10 10 1 n 0
./CUDAthb1024 100000 1000 10 10 1 n 0

./CUDAthb32 100000 1000 10 10 1 p 0
./CUDAthb64 100000 1000 10 10 1 p 0
./CUDAthb128 100000 1000 10 10 1 p 0
./CUDAthb256 100000 1000 10 10 1 p 0
./CUDAthb512 100000 1000 10 10 1 p 0
./CUDAthb1024 100000 1000 10 10 1 p 0

./CUDAthb32 100000 1000 10 10 1 t 0
./CUDAthb64 100000 1000 10 10 1 t 0
./CUDAthb128 100000 1000 10 10 1 t 0
./CUDAthb256 100000 1000 10 10 1 t 0
./CUDAthb512 100000 1000 10 10 1 t 0
./CUDAthb1024 100000 1000 10 10 1 t 0

./CUDAthb32 100000 1000 10 10 1 c 0
./CUDAthb64 100000 1000 10 10 1 c 0
./CUDAthb128 100000 1000 10 10 1 c 0
./CUDAthb256 100000 1000 10 10 1 c 0
./CUDAthb512 100000 1000 10 10 1 c 0
./CUDAthb1024 100000 1000 10 10 1 c 0

# running tests on num Queries on CUDA with 6 testing values

# running test on threads number with query paral in OMP

#./CPPtest_nt 100000 1000 10 10 12 1 p1 0

# running test on len seq on privatization

#./CPPtest_lenSeq_P 50000 1000 10 10 9 12 p1 0


# TODO running test on len seq for thread with all the methods
#./CPPtest_lenSeq 50000 1000 10 3 9 12 s 0
#./CPPtest_lenSeq 50000 1000 10 10 9 12 p1 0
