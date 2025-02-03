## Compilation
Compile files using `make` command.

## Executables
- Sequential version : main
- Parallelized version (OpenMP) : main_OMP
- Parallelized++ version (OpenMP + MPI) : main_OMP_MPI

## Run commands
Note that the executable file has arguments, input filename and output filename, thus:  `./exec_name <input filename> [<output filename>]`. The default Output is the standard output.
- Sequential version :
    - Like any normal executable binary : `./main`.
- Parallelized version:
    - Similar to the sequential version. 
        - Note that you can explicitly set the number of threads inside your C code using `omp_set_num_threads()` Or you can set
        the Number of Threads Using an Environment Variable `export OMP_NUM_THREADS=(number of threads that you want)`.
- Parallelized++ version:
    - Since we use the compilor of MPI `mpicc` to compile this version, running it requires `mpirun` command:
    
        - To run this version naively (1 thread - 1 process):
        ```bash
        mpirun ./exec_name .
        ```
        - If you want to run this version across multiple machines (nodes) listed in a file `host.txt` use :
        ```bash
        mpirun mpirun -hostfile host.txt
        ```
        example of a host.txt file :
        ```
        node1 slots=4
        node2 slots=2
        node3 slots=4
        ...
        ```
        node1 will run 4 MPI processes and node2 will run 2 MPI processes etc.

        - Or better use:
        ```bash
        mpirun -hostfile host.txt --map-by ppr:(number of threads to disribut on the nodes):node ./exec_name
        ``` 
        [Read the documentation for more...](https://docs.open-mpi.org/en/main/man-openmpi/man1/mpirun.1.html)


[More infos and performance tests on input/ data.]()
