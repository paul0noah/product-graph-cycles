message("Retrieving thrust...")
include(FetchContent)
FetchContent_Declare(
    thrust
    GIT_REPOSITORY git@github.com:NVIDIA/thrust.git
    GIT_TAG 756c5af
)
FetchContent_MakeAvailable(thrust)
