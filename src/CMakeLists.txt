# Specify .cpp files
set(TARGET_NAME FoM_Optimizer)

set(SRC FoM_Optimizer.cpp)

# Build an executable 
add_executable(${TARGET_NAME} ${SRC})

# Linker Information
target_link_libraries(${TARGET_NAME} ${ALL_LIBRARIES})

