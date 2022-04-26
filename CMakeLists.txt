cmake_minimum_required(VERSION 3.0)

project(meshimprove)
set(EXE meshimprove)
set(CMAKE_CXX_STANDARD 17)

set(INC_DIR ./inc)
set(HEADER 
${INC_DIR}/equation.h
${INC_DIR}/meshquality.h
${INC_DIR}/NegaOptimize.h
${INC_DIR}/OpenNL_psm.h
${INC_DIR}/smoothing.h
${INC_DIR}/writefile.h
)

set(SRC_DIR ./src)
set(SRC 
${SRC_DIR}/equation.cpp
${SRC_DIR}/mainstart.cc
${SRC_DIR}/meshquality.cpp
${SRC_DIR}/NegaOptimize.cpp
${SRC_DIR}/OpenNL_psm.c
${SRC_DIR}/smoothing.cpp
${SRC_DIR}/writefile.cpp
)

include_directories(./inc)
include_directories(./lib/Eigen)

#include_directories(./lib/OpenVolumeMesh2.0/include)
#link_directories(./lib/OpenVolumeMesh2.0/lib)
include_directories(./lib/OpenVolumeMesh3.0/include)
link_directories(./lib/OpenVolumeMesh3.0/lib)

#include_directories(./lib/OpenMesh7.1/include)
#link_directories(./lib/OpenMesh7.1/lib)
include_directories(./lib/OpenMesh9.0/include)
link_directories(./lib/OpenMesh9.0/lib)

add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -D_LIB -DWIN32 -DEIGEN2_SUPPORT)

add_executable(${EXE} ${SRC} ${HEADER})
target_link_libraries(${EXE} OpenMeshCored OpenMeshToolsd OpenVolumeMesh)