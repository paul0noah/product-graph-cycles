if(TARGET igl::core)
    return()
endif()
message("Retrieving libigl...")
include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG b1bd5b1
)
FetchContent_MakeAvailable(libigl)
