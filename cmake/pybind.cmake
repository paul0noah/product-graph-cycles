include(FetchContent)
message("Retrieving pybind11...")
FetchContent_Declare(
    pybind
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG 6d22dba
)
FetchContent_MakeAvailable(pybind)
