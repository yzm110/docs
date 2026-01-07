project(KalmanLib)
cmake_minimum_required(VERSION 3.0)
set(INCLUDE_DIRECTORIES ./ /usr/include/eigen3)

set(SOURCE_DIRS 
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
foreach(DIR ${SOURCE_DIRS})
    file(GLOB DIR_SOURCES "${DIR}/*.cpp")
    list(APPEND SOURCE_FILE ${DIR_SOURCES})
endforeach()

set(C_FLAGS            "-s")
set(C_FLAGS_DEBUG      "-g")
set(C_FLAGS_RELEASE    "")
set(CXX_FLAGS          "-std=c++17")
set(CXX_FLAGS_DEBUG    "-g -std=c++17")
set(CXX_FLAGS_RELEASE  "-Os -std=c++17")

# build
set(CMAKE_C_FLAGS                "${C_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG          "${C_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_RELEASE        "${C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS              "${CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG        "${CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE      "${CXX_FLAGS_RELEASE}")

include_directories(${INCLUDE_DIRECTORIES})
add_executable(KalmanLibExe ${SOURCE_FILE})