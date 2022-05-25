cmake_minimum_required(VERSION 3.0)

project(meshimprove)

#set(CMAKE_BUILD_TYPE Release)

if(DEFINED CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Set CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
else() 
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Set CMAKE_BUILD_TYPE=Debug")
endif()
message(STATUS "CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})

set(EXE meshimprove)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")

set(INC_DIR ./inc)
set(SRC_DIR ./src)

set(HEADER 
${INC_DIR}/equation.h
${INC_DIR}/meshquality.h
${INC_DIR}/NegaOptimize.h
${INC_DIR}/OpenNL_psm.h
${INC_DIR}/smoothing.h
${INC_DIR}/writefile.h
${INC_DIR}/dataType.h
)

set(SRC 
${SRC_DIR}/equation.cpp
${SRC_DIR}/main_start.cc
${SRC_DIR}/meshquality.cpp
${SRC_DIR}/NegaOptimize.cpp
${SRC_DIR}/OpenNL_psm.c
${SRC_DIR}/smoothing.cpp
${SRC_DIR}/writefile.cpp
)

include_directories(./inc)
include_directories(./lib/Eigen3.4)

include_directories(./lib/OpenVolumeMesh2.0/include)
link_directories(./lib/OpenVolumeMesh2.0/lib)
#include_directories(./lib/OpenVolumeMesh3.0/include)
#link_directories(./lib/OpenVolumeMesh3.0/lib)

include_directories(./lib/OpenMesh7.1/include)
link_directories(./lib/OpenMesh7.1/lib)
#include_directories(./lib/OpenMesh9.0/include)
#link_directories(./lib/OpenMesh9.0/lib)

add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -D_LIB)

add_executable(${EXE} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${EXE} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd)
else()
target_link_libraries(${EXE} OpenMeshCore OpenMeshTools OpenVolumeMesh)
endif()

# NegOPT
set(TARGET_NAME NegOPT)
set(SRC ${SRC_DIR}/main_negopt.cpp ${SRC_DIR}/smoothing.cpp ${SRC_DIR}/meshquality.cpp)
add_executable(${TARGET_NAME} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${TARGET_NAME} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd)
else()
target_link_libraries(${TARGET_NAME} OpenMeshCore OpenMeshTools OpenVolumeMesh)
endif()

#Compare
set(TARGET_NAME Compare)
set(SRC ${SRC_DIR}/main_compare.cpp ${SRC_DIR}/meshquality.cpp)
add_executable(${TARGET_NAME} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${TARGET_NAME} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd)
else()
target_link_libraries(${TARGET_NAME} OpenMeshCore OpenMeshTools OpenVolumeMesh)
endif()

#Time
set(TARGET_NAME Time)
set(SRC ${SRC_DIR}/main_timecal.cpp ${SRC_DIR}/smoothing.cpp ${SRC_DIR}/meshquality.cpp)
add_executable(${TARGET_NAME} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${TARGET_NAME} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd)
else()
target_link_libraries(${TARGET_NAME} OpenMeshCore OpenMeshTools OpenVolumeMesh)
endif()

#quality
set(TARGET_NAME Quality)
set(SRC ${SRC_DIR}/main_quality.cpp ${SRC_DIR}/meshquality.cpp)
add_executable(${TARGET_NAME} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${TARGET_NAME} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd)
else()
target_link_libraries(${TARGET_NAME} OpenMeshCore OpenMeshTools OpenVolumeMesh)
endif()

#UNIT TEST
set(TARGET_NAME UNIT_TEST)
set(SRC ${SRC_DIR}/main_test.cpp)
add_executable(${TARGET_NAME} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${TARGET_NAME} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd)
else()
target_link_libraries(${TARGET_NAME} OpenMeshCore OpenMeshTools OpenVolumeMesh)
endif()

#DRL_SMOOTHING
set(TARGET_NAME DRL_Smoothing)

set(INC_DIR ./inc)
set(SRC_DIR ./src)

set(HEADER 
${INC_DIR}/equation.h
${INC_DIR}/meshquality.h
${INC_DIR}/NegaOptimize.h
${INC_DIR}/OpenNL_psm.h
${INC_DIR}/smoothing.h
${INC_DIR}/writefile.h
${INC_DIR}/dataType.h
)

set(SRC 
${SRC_DIR}/equation.cpp
${SRC_DIR}/main_DRL.cpp
${SRC_DIR}/meshquality.cpp
${SRC_DIR}/NegaOptimize.cpp
${SRC_DIR}/OpenNL_psm.c
${SRC_DIR}/smoothing.cpp
${SRC_DIR}/writefile.cpp
)

set(DRL_DIR ./DRL)
set(SRC ${SRC} 
${DRL_DIR}/cnn_api.cpp
${DRL_DIR}/DeepLearningNetwork.cpp
${DRL_DIR}/evaluatePolicy.cpp
${DRL_DIR}/evaluatePolicy_data.cpp
${DRL_DIR}/evaluatePolicy_initialize.cpp
${DRL_DIR}/evaluatePolicy_rtwutil.cpp
${DRL_DIR}/evaluatePolicy_terminate.cpp
${DRL_DIR}/MWCNNLayerImpl.cpp
${DRL_DIR}/MWMkldnnUtils.cpp
${DRL_DIR}/MWTanhLayer.cpp
${DRL_DIR}/MWTanhLayerImpl.cpp
${DRL_DIR}/MWTargetNetworkImpl.cpp
${DRL_DIR}/predict.cpp
)

include_directories(./inc)
include_directories(./lib/Eigen3.4)

include_directories(./lib/OpenVolumeMesh2.0/include)
link_directories(./lib/OpenVolumeMesh2.0/lib)

include_directories(./lib/OpenMesh7.1/include)
link_directories(./lib/OpenMesh7.1/lib)

include_directories(C:/software/oneDNN/include/)
link_directories(C:/software/oneDNN/bin/)
link_directories(C:/software/oneDNN/lib/)

include_directories(D:/Codes/Mesh_Generation/meshimprove_VS2019/DRL/)
link_directories(D:/Codes/Mesh_Generation/meshimprove_VS2019/DRL/)

add_executable(${TARGET_NAME} ${SRC} ${HEADER})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
target_link_libraries(${TARGET_NAME} OpenMeshCored OpenMeshToolsd OpenVolumeMeshd mkldnn)
else()
target_link_libraries(${TARGET_NAME} OpenMeshCore OpenMeshTools OpenVolumeMesh mkldnn)
endif()