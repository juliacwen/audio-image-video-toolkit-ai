cmake_minimum_required(VERSION 3.10)
project(AudioTools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------
# Build Profile Selection
# ------------------------
set(BUILD_PROFILE "DESKTOP" CACHE STRING "Build profile: DESKTOP, WEARABLE, or EMBEDDED")
set_property(CACHE BUILD_PROFILE PROPERTY STRINGS DESKTOP WEARABLE EMBEDDED)
message(STATUS "Build profile: ${BUILD_PROFILE}")

# ------------------------
# Compiler warnings
# ------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
elseif(MSVC)
  add_compile_options(/W4)
endif()

# ------------------------
# Paths & output
# ------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ------------------------
# Threading support (for tests)
# ------------------------
find_package(Threads REQUIRED)

# ------------------------
# Math library (Unix only)
# ------------------------
if(UNIX AND NOT APPLE)
  set(MATH_LIB m)
elseif(APPLE)
  set(MATH_LIB "")
else()
  set(MATH_LIB "")
endif()

# ------------------------
# Includes used by manual build
# ------------------------
include_directories(
  ${PROJECT_SOURCE_DIR}/inc
  ${PROJECT_SOURCE_DIR}/rnnoise/include
  ${PROJECT_SOURCE_DIR}/src  # Added for utility headers
)

# ------------------------
# RNNoise static lib (explicit path in repo)
# ------------------------
set(RNNOISE_LIB_PATH "${PROJECT_SOURCE_DIR}/rnnoise/.libs/librnnoise.a")
if(EXISTS "${RNNOISE_LIB_PATH}")
  set(RNNOISE_LIB "${RNNOISE_LIB_PATH}")
else()
  find_library(RNNOISE_LIB NAMES rnnoise PATHS ${PROJECT_SOURCE_DIR}/rnnoise/.libs ${PROJECT_SOURCE_DIR}/rnnoise)
endif()

if(RNNOISE_LIB)
  message(STATUS "RNNoise: ${RNNOISE_LIB}")
else()
  message(WARNING "RNNoise library not found - live_audio_denoise may not link correctly")
endif()

# ------------------------
# PortAudio: robust find (path or pkg-config)
# ------------------------
set(PORTAUDIO_INCLUDE_DIR "" CACHE PATH "PortAudio include dir (optional)")
set(PORTAUDIO_LIBRARY "" CACHE PATH "PortAudio library path (optional)")

function(resolve_portaudio out_inc out_lib)
  if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARY)
    set(${out_inc} ${PORTAUDIO_INCLUDE_DIR} PARENT_SCOPE)
    set(${out_lib} ${PORTAUDIO_LIBRARY} PARENT_SCOPE)
    return()
  endif()

  # try find_path/find_library (common Homebrew/include locations)
  find_path(_pa_inc portaudio.h
    PATHS /opt/homebrew/include /usr/local/include /usr/include NO_DEFAULT_PATH)
  find_library(_pa_lib NAMES portaudio libportaudio
    PATHS /opt/homebrew/lib /usr/local/lib /usr/lib NO_DEFAULT_PATH)

  if(_pa_inc AND _pa_lib)
    set(${out_inc} ${_pa_inc} PARENT_SCOPE)
    set(${out_lib} ${_pa_lib} PARENT_SCOPE)
    return()
  endif()

  # fallback: try pkg-config parsing
  find_program(_PKGCFG pkg-config)
  if(_PKGCFG)
    execute_process(COMMAND ${_PKGCFG} --cflags portaudio-2.0
      OUTPUT_VARIABLE _pc_cflags OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    execute_process(COMMAND ${_PKGCFG} --libs portaudio-2.0
      OUTPUT_VARIABLE _pc_libs OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_pc_cflags)
      string(REGEX MATCHALL "-I[^ ]+" _incs "${_pc_cflags}")
      foreach(i ${_incs})
        string(REGEX REPLACE "^-I" "" _p "${i}")
        if(EXISTS "${_p}")
          set(${out_inc} ${_p} PARENT_SCOPE)
          break()
        endif()
      endforeach()
    endif()
    if(_pc_libs)
      # try to find absolute lib path in the output (-L paths or absolute libs)
      string(REGEX MATCHALL "/[^ ]+portaudio[^ ]*" _abslibs "${_pc_libs}")
      if(_abslibs)
        list(GET _abslibs 0 _abs)
        if(EXISTS "${_abs}")
          set(${out_lib} ${_abs} PARENT_SCOPE)
          return()
        endif()
      endif()
      # otherwise try to extract -L dirs then find lib in them
      string(REGEX MATCHALL "-L[^ ]+" _ldirs "${_pc_libs}")
      foreach(d ${_ldirs})
        string(REGEX REPLACE "^-L" "" _dir "${d}")
        file(GLOB _cand "${_dir}/libportaudio.*")
        if(_cand)
          list(GET _cand 0 _chosen)
          set(${out_lib} ${_chosen} PARENT_SCOPE)
          return()
        endif()
      endforeach()
    endif()
  endif()

  set(${out_inc} "" PARENT_SCOPE)
  set(${out_lib} "" PARENT_SCOPE)
endfunction()

resolve_portaudio(PORTAUDIO_INCLUDE_DIR_FOUND PORTAUDIO_LIBRARY_FOUND)
if(NOT PORTAUDIO_INCLUDE_DIR_FOUND OR NOT PORTAUDIO_LIBRARY_FOUND)
  message(FATAL_ERROR "PortAudio not found. Either install PortAudio or set PORTAUDIO_INCLUDE_DIR and PORTAUDIO_LIBRARY cache variables.")
endif()

message(STATUS "PortAudio include: ${PORTAUDIO_INCLUDE_DIR_FOUND}")
message(STATUS "PortAudio lib: ${PORTAUDIO_LIBRARY_FOUND}")

# ------------------------
# Shared Utility Libraries
# ------------------------
# WAV processing utilities
add_library(wav_utils STATIC
  src/wav_utils.cpp
)
target_include_directories(wav_utils PUBLIC ${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/src)

# FFT utilities
add_library(fft_utils STATIC
  src/fft_utils.cpp
)
target_include_directories(fft_utils PUBLIC ${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/src)

# ------------------------
# Live Audio Denoise Executables (Multiple Profiles)
# ------------------------
# Common source files
set(DENOISE_SOURCES
  src/audio_framework_base.cpp
  src/live_audio_denoise.cpp
  src/wav_writer.cpp
)

# Desktop build (full features)
add_executable(live_audio_denoise_desktop ${DENOISE_SOURCES})
target_compile_definitions(live_audio_denoise_desktop PRIVATE BUILD_DESKTOP)
target_include_directories(live_audio_denoise_desktop PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_denoise_desktop PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)
if(RNNOISE_LIB)
  target_link_libraries(live_audio_denoise_desktop PRIVATE ${RNNOISE_LIB})
endif()

# Wearable build (power optimized)
add_executable(live_audio_denoise_wearable ${DENOISE_SOURCES})
target_compile_definitions(live_audio_denoise_wearable PRIVATE BUILD_WEARABLE)
target_include_directories(live_audio_denoise_wearable PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_denoise_wearable PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)
if(RNNOISE_LIB)
  target_link_libraries(live_audio_denoise_wearable PRIVATE ${RNNOISE_LIB})
endif()

# Embedded build (minimal resources)
add_executable(live_audio_denoise_embedded ${DENOISE_SOURCES})
target_compile_definitions(live_audio_denoise_embedded PRIVATE BUILD_EMBEDDED)
target_include_directories(live_audio_denoise_embedded PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_denoise_embedded PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)
if(RNNOISE_LIB)
  target_link_libraries(live_audio_denoise_embedded PRIVATE ${RNNOISE_LIB})
endif()

# ------------------------
# Live Audio Tonal Detection Executables (Multiple Profiles)
# ------------------------
# Common source files
set(TONAL_SOURCES
  src/audio_framework_base.cpp
  src/live_audio_tonal_detection.cpp
  src/wav_writer.cpp
)

# Desktop build (full features)
add_executable(live_audio_tonal_detection_desktop ${TONAL_SOURCES})
target_compile_definitions(live_audio_tonal_detection_desktop PRIVATE BUILD_DESKTOP)
target_include_directories(live_audio_tonal_detection_desktop PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_tonal_detection_desktop PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)

# Wearable build (power optimized)
add_executable(live_audio_tonal_detection_wearable ${TONAL_SOURCES})
target_compile_definitions(live_audio_tonal_detection_wearable PRIVATE BUILD_WEARABLE)
target_include_directories(live_audio_tonal_detection_wearable PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_tonal_detection_wearable PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)

# Embedded build (minimal resources)
add_executable(live_audio_tonal_detection_embedded ${TONAL_SOURCES})
target_compile_definitions(live_audio_tonal_detection_embedded PRIVATE BUILD_EMBEDDED)
target_include_directories(live_audio_tonal_detection_embedded PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_tonal_detection_embedded PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)


# ------------------------
# Live Audio Denoise Network Executables (Multiple Profiles)
# ------------------------
# Network version source files
set(DENOISE_NETWORK_SOURCES
  src/live_audio_denoise_network.cpp
  src/audio_jitter_buffer.cpp
  src/audio_io_context.cpp
  src/network_rtp.cpp
  src/wav_writer.cpp
)

# Desktop network build
add_executable(live_audio_denoise_network_desktop ${DENOISE_NETWORK_SOURCES})
target_compile_definitions(live_audio_denoise_network_desktop PRIVATE BUILD_DESKTOP)
target_include_directories(live_audio_denoise_network_desktop PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_denoise_network_desktop PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)
if(RNNOISE_LIB)
  target_link_libraries(live_audio_denoise_network_desktop PRIVATE ${RNNOISE_LIB})
endif()

# Wearable network build
add_executable(live_audio_denoise_network_wearable ${DENOISE_NETWORK_SOURCES})
target_compile_definitions(live_audio_denoise_network_wearable PRIVATE BUILD_WEARABLE)
target_include_directories(live_audio_denoise_network_wearable PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_denoise_network_wearable PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)
if(RNNOISE_LIB)
  target_link_libraries(live_audio_denoise_network_wearable PRIVATE ${RNNOISE_LIB})
endif()

# Embedded network build
add_executable(live_audio_denoise_network_embedded ${DENOISE_NETWORK_SOURCES})
target_compile_definitions(live_audio_denoise_network_embedded PRIVATE BUILD_EMBEDDED)
target_include_directories(live_audio_denoise_network_embedded PRIVATE ${PORTAUDIO_INCLUDE_DIR_FOUND})
target_link_libraries(live_audio_denoise_network_embedded PRIVATE 
  ${PORTAUDIO_LIBRARY_FOUND} 
  ${MATH_LIB}
  Threads::Threads
)
if(RNNOISE_LIB)
  target_link_libraries(live_audio_denoise_network_embedded PRIVATE ${RNNOISE_LIB})
endif()

# Create a default alias based on BUILD_PROFILE
if(BUILD_PROFILE STREQUAL "WEARABLE")
  add_custom_target(live_audio_denoise ALL DEPENDS live_audio_denoise_wearable)
  message(STATUS "Default target 'live_audio_denoise' -> wearable profile")
elseif(BUILD_PROFILE STREQUAL "EMBEDDED")
  add_custom_target(live_audio_denoise ALL DEPENDS live_audio_denoise_embedded)
  message(STATUS "Default target 'live_audio_denoise' -> embedded profile")
else()
  add_custom_target(live_audio_denoise ALL DEPENDS live_audio_denoise_desktop)
  message(STATUS "Default target 'live_audio_denoise' -> desktop profile")
endif()

# ------------------------
# Other Executables
# ------------------------
# wav_to_csv - simple WAV to CSV converter
add_executable(wav_to_csv src/wav_to_csv.cpp)
target_link_libraries(wav_to_csv PRIVATE wav_utils)

# wav_freq_csv - single channel with FFT
add_executable(wav_freq_csv src/wav_freq_csv.cpp)
target_link_libraries(wav_freq_csv PRIVATE wav_utils fft_utils)

# wav_freq_csv_channelized - multi-channel with FFT
add_executable(wav_freq_csv_channelized src/wav_freq_csv_channelized.cpp)
target_link_libraries(wav_freq_csv_channelized PRIVATE wav_utils fft_utils)

# ------------------------
# GoogleTest find & robust imported-target creation
# ------------------------
find_package(GTest QUIET)

if(GTest_FOUND)
  message(STATUS "Found GTest via find_package")
else()
  # Try Homebrew locations
  find_path(_gtest_inc gtest/gtest.h PATHS /opt/homebrew/include /usr/local/include)
  find_library(_gtest_lib gtest PATHS /opt/homebrew/lib /usr/local/lib)
  find_library(_gtest_main_lib gtest_main PATHS /opt/homebrew/lib /usr/local/lib)
  if(_gtest_inc AND _gtest_lib AND _gtest_main_lib)
    # create imported targets GTest::GTest and GTest::Main so target_link_libraries works
    add_library(GTest::GTest UNKNOWN IMPORTED)
    set_target_properties(GTest::GTest PROPERTIES
      IMPORTED_LOCATION "${_gtest_lib}"
      INTERFACE_INCLUDE_DIRECTORIES "${_gtest_inc}"
    )
    add_library(GTest::Main UNKNOWN IMPORTED)
    set_target_properties(GTest::Main PROPERTIES
      IMPORTED_LOCATION "${_gtest_main_lib}"
      INTERFACE_INCLUDE_DIRECTORIES "${_gtest_inc}"
      INTERFACE_LINK_LIBRARIES "GTest::GTest"
    )
    set(GTest_FOUND TRUE)
    message(STATUS "Created imported GTest targets from Homebrew locations")
  else()
    message(WARNING "GoogleTest not found. Tests will be skipped.")
    set(SKIP_TESTS TRUE)
  endif()
endif()

if(NOT SKIP_TESTS)
  enable_testing()

  # ------------------------
  # Test executables (they call the utility exes)
  # ------------------------
  add_executable(test_wav_freq_csv tests/test_wav_freq_csv.cpp)
  target_link_libraries(test_wav_freq_csv PRIVATE GTest::GTest GTest::Main Threads::Threads)
  add_test(NAME test_wav_freq_csv COMMAND test_wav_freq_csv)
  set_tests_properties(test_wav_freq_csv PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  add_executable(test_wav_freq_csv_multi_channel tests/test_wav_freq_csv_multi_channel.cpp)
  target_link_libraries(test_wav_freq_csv_multi_channel PRIVATE GTest::GTest GTest::Main Threads::Threads)
  add_test(NAME test_wav_freq_csv_multi_channel COMMAND test_wav_freq_csv_multi_channel)
  set_tests_properties(test_wav_freq_csv_multi_channel PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  add_executable(test_wav_to_csv tests/test_wav_to_csv.cpp)
  target_link_libraries(test_wav_to_csv PRIVATE GTest::GTest GTest::Main Threads::Threads)
  add_test(NAME test_wav_to_csv COMMAND test_wav_to_csv)
  set_tests_properties(test_wav_to_csv PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # ------------------------
  # Unit test for FFT utils
  # ------------------------
  add_executable(test_fft_utils tests/test_fft_utils.cpp src/fft_utils.cpp)
  target_link_libraries(test_fft_utils PRIVATE GTest::GTest GTest::Main Threads::Threads)
  add_test(NAME test_fft_utils COMMAND test_fft_utils)
  set_tests_properties(test_fft_utils PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # ------------------------
  # Unit test for Wav utils
  # ------------------------
  add_executable(test_wav_utils tests/test_wav_utils.cpp src/wav_utils.cpp)
  target_link_libraries(test_wav_utils PRIVATE GTest::GTest GTest::Main Threads::Threads)
  add_test(NAME test_wav_utils COMMAND test_wav_utils)
  set_tests_properties(test_wav_utils PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # ------------------------
  # Unit test for WavWriter utility class
  # ------------------------
  add_executable(test_wav_writer tests/test_wav_writer.cpp src/wav_writer.cpp)
  target_link_libraries(test_wav_writer PRIVATE GTest::GTest GTest::Main Threads::Threads)
  add_test(NAME test_wav_writer COMMAND test_wav_writer)
  set_tests_properties(test_wav_writer PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # ------------------------
  # Unit test for Live Audio Denoise (tests desktop build)
  # ------------------------
  add_executable(test_live_audio_denoise 
      tests/test_live_audio_denoise.cpp
      src/wav_writer.cpp
  )
  target_compile_definitions(test_live_audio_denoise PRIVATE BUILD_DESKTOP)
  target_link_libraries(test_live_audio_denoise 
      PRIVATE 
      GTest::GTest 
      GTest::Main 
      Threads::Threads
      ${MATH_LIB}
      ${PORTAUDIO_LIBRARY_FOUND} 
      ${RNNOISE_LIB}
  )
  add_test(NAME test_live_audio_denoise COMMAND test_live_audio_denoise)
  set_tests_properties(test_live_audio_denoise PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  # ------------------------
  # Unit test for Wearable Profile
  # ------------------------
  add_executable(test_live_audio_denoise_wearable 
      tests/test_live_audio_denoise.cpp
      src/wav_writer.cpp
  )
  target_compile_definitions(test_live_audio_denoise_wearable PRIVATE BUILD_WEARABLE)
  target_link_libraries(test_live_audio_denoise_wearable 
      PRIVATE 
      GTest::GTest 
      GTest::Main 
      Threads::Threads
      ${MATH_LIB}
      ${PORTAUDIO_LIBRARY_FOUND} 
      ${RNNOISE_LIB}
  )
  add_test(NAME test_live_audio_denoise_wearable COMMAND test_live_audio_denoise_wearable)
  set_tests_properties(test_live_audio_denoise_wearable PROPERTIES WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

# ------------------------
# Summary messages
# ------------------------
message(STATUS "")
message(STATUS "=== AudioTools Build Configuration ===")
message(STATUS "Build profile: ${BUILD_PROFILE}")
message(STATUS "Utility libraries: wav_utils, fft_utils")
message(STATUS "")
message(STATUS "Live audio denoise targets:")
message(STATUS "  - live_audio_denoise_desktop  (16ch, 1s buffer, full features)")
message(STATUS "  - live_audio_denoise_wearable (2ch, 200ms buffer, VAD, low power)")
message(STATUS "  - live_audio_denoise_embedded (1ch, 16kHz, minimal resources)")
message(STATUS "  - live_audio_denoise (alias to ${BUILD_PROFILE} profile)")
message(STATUS "")
message(STATUS "Live audio denoise network targets:")
message(STATUS "  - live_audio_denoise_network_desktop")
message(STATUS "  - live_audio_denoise_network_wearable")
message(STATUS "  - live_audio_denoise_network_embedded")
message(STATUS "")
message(STATUS "Other targets: wav_freq_csv, wav_freq_csv_channelized, wav_to_csv")
if(NOT SKIP_TESTS)
  message(STATUS "Tests: enabled")
else()
  message(STATUS "Tests: skipped (GTest not found)")
endif()
message(STATUS "======================================")
message(STATUS "")
