![University of Lincoln](http://thelincolnite.co.uk/wp-content/uploads/2012/07/new_uni_crest.jpg "University of Lincoln")
----------

# CMP3110M Parallel Computing, Assessment Item 1
Your task is to develop a simple statistical tool for analysing historical weather records from
Lincolnshire. The provided data files include records of air temperature collected over a period of
more than 80 years from five weather stations in Lincolnshire: Barkston Heath, Scampton,
Waddington, Cranwell and Coningsby. Your tool should be able to load the provided dataset and
perform statistical summaries of temperature including the min, max and average values, and standard
deviation. The provided summaries should be performed on the entire dataset regardless their
acquisition time and location. For additional credit, you can also consider the median statistic and its
1st and 3rd quartiles (i.e. 25th and 75th percentiles) which will require a development of a suitable
sorting algorithm.

Due to the large amount of data (i.e. 1.8 million records), all statistical calculations shall be performed
on parallel hardware and implemented by a parallel software component written in OpenCL. Your tool
should also report memory transfer, kernel execution and total program execution times for
performance assessment. Further credit will be given for additional optimisation strategies which
target the parallel performance of the tool. In such a case, your program should run and display
execution times for different variants of your algorithm. Your basic implementation can assume
temperature values expressed as integers and skip all parts after a decimal point. For additional credit,
you should also consider the exact temperature values and their corresponding statistics.

You can base your code on the material provided during workshop sessions, but you are not allowed to
use any existing parallel libraries (e.g. Boost.Compute). To help you with code development, a shorter
dataset is also provided which is 100 times smaller. The original file is called
“weather_lincolnshire.txt” and the short dataset is “weather_lincolnshire_short.txt”. More details
about the file format are included in the “readme.txt” file. The data files are provided on Blackboard
together with this description document in a file “temp_lincolnshire_datasets.zip”. The output results
and performance measures should be reported in a console window in a clear and readable format. All
reading and displaying operations should be provided by the host code.

The main assessment criteria for this task are related to the correctness of the developed algorithms
and effectiveness of optimisation strategies. The code should be well commented and clearly structured into functional blocks.

----------


## Objectives


* [LO1] demonstrate practical skills in applying parallel algorithms for solving computational
problems;
* [LO3] analyse parallel architectures as a means to provide solutions to complex computational
problems.

----------


## License

The program is licenced under [GPL Version 2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html). The source is freely available to use, compile and modify.


----------
