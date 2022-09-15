# cmake file for pybind11 project. the project name must be the exact same as the .cpp file that wraps it into python.

cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(rolling_statistics_py)

file (GLOB FILES "src/*.cpp" "src/*.hpp")

# Set up such that XCode organizes the files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${FILES} )

find_package(pybind11 REQUIRED)

pybind11_add_module(rolling_statistics_py
	${FILES}
)

target_link_libraries(rolling_statistics_py PUBLIC)

install(TARGETS rolling_statistics_py
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
  )
