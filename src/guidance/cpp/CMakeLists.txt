cmake_minimum_required(VERSION 3.15)
project(missile_pathfinder)

# --- 1. Settings ---
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------
# [STEP 1] FORCE SPECIFIC PYTHON PATH
# REPLACE THE PATH BELOW WITH THE ONE YOU COPIED FROM TERMINAL
set(Python3_EXECUTABLE "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3")
# -----------------------------------------------------------

# --- 2. Find Python ---
# We use "EXACT" to force it to respect the executable we set above
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# --- 3. SMART LOCATOR: Ask YOUR Python where pybind11 is ---
# We use the found Python executable to ask for the path.
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_status
)

if("${_pybind11_status}" EQUAL 0 AND NOT "${_pybind11_path}" STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_path}")
    message(STATUS "Pybind11 found at: ${_pybind11_path}")
else()
    message(WARNING "Could not auto-detect pybind11. Ensure 'pip install pybind11' is run for ${Python3_EXECUTABLE}")
endif()

# --- 4. Now Find Pybind11 ---
find_package(pybind11 REQUIRED)

# --- 5. Create the Module ---
pybind11_add_module(missile_backend pathfinder.cpp)

# --- 6. Optimization ---
if(MSVC)
    target_compile_options(missile_backend PRIVATE /O2 /fp:fast)
else()
    # Mac Optimization flags
    target_compile_options(missile_backend PRIVATE -O3 -ffast-math -march=native)
endif()

# --- 7. Auto-Copy to Source Folder ---
add_custom_command(TARGET missile_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:missile_backend>
        ${CMAKE_CURRENT_SOURCE_DIR}/../
)