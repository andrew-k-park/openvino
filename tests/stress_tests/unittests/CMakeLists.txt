# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "StressUnitTests")

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_link_libraries(${TARGET_NAME} PRIVATE StressTestsCommon)

install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

# Copy local configs to BIN_FOLDER
configure_file(local_configs/test_config.xml
    ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/stress_tests_configs/unittests/test_config.xml COPYONLY)
