# len seq, len query, num queries, n runs, n iterations, n threads, type, verbose

# TODO tonight, March 22 2020

./CPP-executables/CPPlenseq 50000 1000 10 3 9 12 s 0

./CPP-executables/CPPnqueries 100000 1000 10 3 6 12 s 0
./CPP-executables/CPPnqueries 100000 1000 10 10 6 12 pq 0
./CPP-executables/CPPnqueries 100000 1000 10 10 6 12 pd 0
./CPP-executables/CPPnqueries 100000 1000 10 10 6 12 pdl 0

./CPP-executables/CPPlenquery 100000 100 10 10 7 12 pq 0
./CPP-executables/CPPlenquery 100000 100 10 10 7 12 pd 0
./CPP-executables/CPPlenquery 100000 100 10 10 7 12 pdl 0
#
./CPP-executables/CPPlenseq 50000 1000 10 10 9 12 pq 0
./CPP-executables/CPPlenseq 50000 1000 10 10 9 12 pd 0
./CPP-executables/CPPlenseq 50000 1000 10 10 9 12 pdl 0

