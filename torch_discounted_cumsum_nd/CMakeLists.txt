cmake_minimum_required(VERSION 3.23)

project(profile_cumsum LANGUAGES CXX CUDA)

include(CPM.cmake)

cpmaddpackage(
  NAME
  cxxopts
  GITHUB_REPOSITORY
  jarro2783/cxxopts
  VERSION
  3.2.0
  OPTIONS
  "CXXOPTS_BUILD_EXAMPLES NO"
  "CXXOPTS_BUILD_TESTS NO")

# Query python as to where PyTorch is located
execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0])"
                OUTPUT_VARIABLE TORCH_PATH)
string(STRIP ${TORCH_PATH} TORCH_PATH)

find_package(Torch REQUIRED HINTS "${TORCH_PATH}/share/cmake/Torch")

find_package(TBB REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(discount-cumsum-profile operator.cu profile.cpp)

target_compile_options(
  discount-cumsum-profile
  PUBLIC $<$<CONFIG:Debug>:-O0;-g;$<$<COMPILE_LANGUAGE:CUDA>:-G;>>
         $<$<CONFIG:RelWithDebInfo>:-g;-O2;$<$<COMPILE_LANGUAGE:CUDA>:-G;>>
         $<$<CONFIG:Release>:-O2;>)

set_target_properties(discount-cumsum-profile PROPERTIES CUDA_ARCHITECTURES
                                                         "native")

target_link_libraries(discount-cumsum-profile PUBLIC ${TORCH_LIBRARIES}
                                                     TBB::tbb cxxopts::cxxopts)
