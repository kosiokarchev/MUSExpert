cmake_minimum_required(VERSION 3.13)
project(incubator)

enable_language(Fortran)
set(CMAKE_C_COMPILER gcc-8)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_Fortran_COMPILER gfortran)

set(CMAKE_C_STANDARD 11)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
message("Python: " ${Python3_EXECUTABLE})
message("Python include: " ${Python3_INCLUDE_DIRS} )
message("Python libs: " ${Python3_LIBRARIES} )

if(NOT Python3_NumPy_FOUND)
    execute_process(COMMAND
            "${Python3_EXECUTABLE}" -c
            "try: import numpy; print(numpy.get_include(), end='')\nexcept:pass\n"
            OUTPUT_VARIABLE Python3_NumPy_INCLUDE_DIRS)
endif()
message("NumPy: " ${Python3_NumPy_INCLUDE_DIRS})

find_library(CFITSIO_LIBRARY NAMES cfitsio)
message("cfitsio: " ${CFITSIO_LIBRARY} )

include_directories("/usr/local/include")
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Python3_NumPy_INCLUDE_DIRS})

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fbackslash -cpp -O3 -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans -fopenmp")

set(SOURCE_FILES incubator.c cube.c fortran.h funcs.f95 incubator.f95)
add_library(incubator SHARED ${SOURCE_FILES})

set_target_properties(incubator PROPERTIES LINKER_LANGUAGE Fortran)
target_link_libraries(incubator ${Python3_LIBRARIES})
target_link_libraries(incubator ${CFITSIO_LIBRARY})
