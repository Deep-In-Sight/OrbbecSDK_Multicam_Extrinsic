cmake_minimum_required(VERSION 3.1.15)
project(Mean_pcd)

# C++ 표준 설정
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OrbbecSDK 설정
set(OBSensor_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../")

# 라이브러리 디렉토리 설정 (시스템에 따라 다름)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(OBSensor_LIBRARY_DIRS "${OBSensor_ROOT_DIR}/lib/linux_x64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(OBSensor_LIBRARY_DIRS "${OBSensor_ROOT_DIR}/lib/arm64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(OBSensor_LIBRARY_DIRS "${OBSensor_ROOT_DIR}/lib/arm32")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(OBSensor_LIBRARY_DIRS "${OBSensor_ROOT_DIR}/lib/win_x64")
    else()
        set(OBSensor_LIBRARY_DIRS "${OBSensor_ROOT_DIR}/lib/win_x86")
    endif()
endif()

# 헤더 파일 경로
include_directories("${OBSensor_ROOT_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../")

# 라이브러리 경로
link_directories("${OBSensor_LIBRARY_DIRS}")

# OpenCV 찾기
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Eigen 찾기
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

# 실행 파일 생성
add_executable(${PROJECT_NAME} Mean_pcd.cpp)

target_sources(Mean_pcd
                PRIVATE
                Mean_pcd.cpp
                ../Sample-MultiDeviceSync/cJSON.c
)

target_include_directories(Mean_pcd
                PRIVATE
                ../Sample-MultiDeviceSync/
)

# 라이브러리 링크
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME}
        -lOrbbecSDK
        ${OpenCV_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
        ${APRILTAG_LIBRARIES}
        -lpthread
        -lm
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(${PROJECT_NAME}
        OrbbecSDK
        ${OpenCV_LIBS}
    )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(${PROJECT_NAME}
        -lOrbbecSDK
        ${OpenCV_LIBS}
        -lpthread
        -lm
    )
endif()

# RPATH 설정 (Linux와 macOS용)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "${OBSensor_LIBRARY_DIRS}"
        INSTALL_RPATH "${OBSensor_LIBRARY_DIRS}"
    )
endif()

# 실행 파일을 빌드 디렉토리로 복사
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)