cmake_minimum_required(VERSION 3.15)
project(minah LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

# Find dependencies
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED)
find_package(ZeroMQ REQUIRED)
find_package(PkgConfig REQUIRED)

# Find WebSocket++
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)

# Find web3cpp (clone if not found)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/web3cpp")
    execute_process(
        COMMAND git clone --recursive https://github.com/ethereum/web3cpp.git third_party/web3cpp
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()

# Add web3cpp as subdirectory
add_subdirectory(third_party/web3cpp)

# Find Fix8 (adjust these paths as needed)
find_path(FIX8_INCLUDE_DIR NAMES fix8/f8includes.hpp)
find_library(FIX8_LIBRARY NAMES Fix8)

if(NOT FIX8_INCLUDE_DIR)
    message(FATAL_ERROR "Fix8 include directory not found. Please set FIX8_INCLUDE_DIR.")
endif()
if(NOT FIX8_LIBRARY)
    message(FATAL_ERROR "Fix8 library not found. Please set FIX8_LIBRARY.")
endif()

include_directories(${FIX8_INCLUDE_DIR})

# Collect all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Collect blockchain source files separately
file(GLOB_RECURSE BLOCKCHAIN_SOURCES "src/blockchain/*.cpp")

# Add blockchain include directories
include_directories(${CMAKE_SOURCE_DIR}/src/blockchain)
include_directories(${CMAKE_SOURCE_DIR}/src/blockchain/utils)
include_directories(${CMAKE_SOURCE_DIR}/src/blockchain/connectors)
include_directories(${CMAKE_SOURCE_DIR}/src/blockchain/adapters)
include_directories(${CMAKE_SOURCE_DIR}/src/blockchain/pipeline)

# Executable
add_executable(minah ${SOURCES})

# Blockchain test executable
add_executable(blockchain_integration_test tests/blockchain_integration_test.cpp ${BLOCKCHAIN_SOURCES})
target_link_libraries(blockchain_integration_test
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    web3cpp
    pthread
)

# Link libraries
target_link_libraries(minah
    yaml-cpp
    Boost::boost
    ${ZeroMQ_LIBRARIES}
    ${FIX8_LIBRARY}
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    web3cpp
)

# Install rule
install(TARGETS minah DESTINATION bin)