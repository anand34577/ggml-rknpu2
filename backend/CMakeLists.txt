# HINTS, not PATHS: PATHS are searched after the system dirs, so an older librknnrt.so
# in /usr/lib (common on vendor images) would silently win over the one asked for
find_library(RKNNRT_LIBRARY
    NAMES rknnrt
    HINTS ${GGML_RKNPU2_LIB_DIR}
    PATHS /usr/lib /usr/lib64 /usr/lib/aarch64-linux-gnu /usr/local/lib)

find_path(RKNNRT_INCLUDE_DIR
    NAMES rknn_matmul_api.h
    HINTS ${GGML_RKNPU2_INCLUDE_DIR}
    PATHS /usr/include /usr/local/include)

if (NOT RKNNRT_LIBRARY OR NOT RKNNRT_INCLUDE_DIR)
    message(FATAL_ERROR "RKNPU2: librknnrt.so or rknn_matmul_api.h not found. "
        "Install them from https://github.com/airockchip/rknn-toolkit2 (rknpu2/runtime) "
        "or set -DGGML_RKNPU2_LIB_DIR=... -DGGML_RKNPU2_INCLUDE_DIR=...")
endif()

message(STATUS "RKNPU2: found ${RKNNRT_LIBRARY}")

# an imported target lets CMake pass the library dir on (-rpath-link) to everything that
# links ggml-rknpu2, a plain file path breaks the link when the runtime is outside /usr/lib
if (RKNNRT_LIBRARY MATCHES "\\.a$")
    add_library(rknnrt STATIC IMPORTED)
else()
    add_library(rknnrt SHARED IMPORTED)
endif()
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION             ${RKNNRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${RKNNRT_INCLUDE_DIR})

ggml_add_backend_library(ggml-rknpu2
                         ggml-rknpu2.cpp
                        )

target_link_libraries(ggml-rknpu2 PRIVATE rknnrt)
