# Pattern Recognition

This repository is a collaboration with [Giulio Bazzanti](https://github.com/giuliobz) as the final project for the [Parallel Computing](https://www.unifi.it/p-ins2-2015-415640-0.html) exam at the University of Florence. The mid-term project, DES-decrypting, can be found in [this repository](https://github.com/giuliobz/DES-Decrypting).

Here we propose our solutions for the Pattern Recognition task. In particular, we study the CPU and GPU parallelism, through OpenMP and CUDA implementations respectively. We compare those performances with a sequential method and we evaluate the resulting Speed Up and Efficiency.

For more theorical details of our project please refers to the review(**IN PROGRESS**) and our presentation(**TODO**).

## Reproduce experiments
### Install  the repository
* `$ git clone git@github.com:NiccoBiondi/Pattern_Recognition.git`
* `$ cd Pattern_Recognition`

### Run the experiments
There are a list of executable in the test/(**TODO**) that we create to produce our experiments and an example of a bash script useful to run jointly multiple experiments.

Anyway you can generate you own executable both trough the [CMake File](https://github.com/NiccoBiondi/Pattern_Recognition/blob/master/CMakeLists.txt) or as follows: 

* `$ g++ ...`

This one generate the executable for sequential and CPU parallelism and it requires the following arguments: 

| Parameter | Type | Description | 
| ------ | ------ | ------ |
| len_seq | int | Historical data length |
| len_query | int | Query length |
| num_queries | int | Total queries number |
| runs | int | Number of same execution for computing mean and std |
| iterations | int | Total testing iterations number |
| nthreads | int | Threads number |
| type | char | Specify the execution type (can be **s**, **pq** or **pd**) |
| verbose | int | Useful print (can be **0**, **1** or **2**) |

* `$ nvcc ...`

The resulting CUDA executable requires:

| Parameter | Type | Description | 
| ------ | ------ | ------ |
| len_seq | int | Historical data length |
| len_query | int | Query length |
| num_queries | int | Total queries number |
| runs | int | Number of same execution for computing mean and std |
| iterations | int | Total testing iterations number |
| type | char | Specify the execution type (can be **n**, **p**, **t** or **c**) |
| verbose | int | Useful print (can be **0**, **1** or **2**) |
