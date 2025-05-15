message("Retrieving robin-map...")
include(FetchContent)
FetchContent_Declare(
    robin-map
    GIT_REPOSITORY https://github.com/Tessil/robin-map.git
    GIT_TAG 5eace6f74c9edff8e264c2d26a85365ad9ea149c
)
FetchContent_MakeAvailable(robin-map)