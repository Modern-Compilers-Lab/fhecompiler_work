# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aimad/Desktop/projects/fhe_compiler/fheco_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aimad/Desktop/projects/fhe_compiler/fheco_project

# Include any dependencies generated for this target.
include CMakeFiles/dsl.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/dsl.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/dsl.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dsl.dir/flags.make

CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o: CMakeFiles/dsl.dir/flags.make
CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o: src/dsl/ciphertext.cpp
CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o: CMakeFiles/dsl.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aimad/Desktop/projects/fhe_compiler/fheco_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o -MF CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o.d -o CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o -c /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/ciphertext.cpp

CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/ciphertext.cpp > CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.i

CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/ciphertext.cpp -o CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.s

CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o: CMakeFiles/dsl.dir/flags.make
CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o: src/dsl/plaintext.cpp
CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o: CMakeFiles/dsl.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aimad/Desktop/projects/fhe_compiler/fheco_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o -MF CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o.d -o CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o -c /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/plaintext.cpp

CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/plaintext.cpp > CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.i

CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/plaintext.cpp -o CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.s

CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o: CMakeFiles/dsl.dir/flags.make
CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o: src/dsl/scalar.cpp
CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o: CMakeFiles/dsl.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aimad/Desktop/projects/fhe_compiler/fheco_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o -MF CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o.d -o CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o -c /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/scalar.cpp

CMakeFiles/dsl.dir/src/dsl/scalar.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dsl.dir/src/dsl/scalar.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/scalar.cpp > CMakeFiles/dsl.dir/src/dsl/scalar.cpp.i

CMakeFiles/dsl.dir/src/dsl/scalar.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dsl.dir/src/dsl/scalar.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aimad/Desktop/projects/fhe_compiler/fheco_project/src/dsl/scalar.cpp -o CMakeFiles/dsl.dir/src/dsl/scalar.cpp.s

# Object files for target dsl
dsl_OBJECTS = \
"CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o" \
"CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o" \
"CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o"

# External object files for target dsl
dsl_EXTERNAL_OBJECTS =

libdsl.a: CMakeFiles/dsl.dir/src/dsl/ciphertext.cpp.o
libdsl.a: CMakeFiles/dsl.dir/src/dsl/plaintext.cpp.o
libdsl.a: CMakeFiles/dsl.dir/src/dsl/scalar.cpp.o
libdsl.a: CMakeFiles/dsl.dir/build.make
libdsl.a: CMakeFiles/dsl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aimad/Desktop/projects/fhe_compiler/fheco_project/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libdsl.a"
	$(CMAKE_COMMAND) -P CMakeFiles/dsl.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dsl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dsl.dir/build: libdsl.a
.PHONY : CMakeFiles/dsl.dir/build

CMakeFiles/dsl.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dsl.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dsl.dir/clean

CMakeFiles/dsl.dir/depend:
	cd /home/aimad/Desktop/projects/fhe_compiler/fheco_project && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aimad/Desktop/projects/fhe_compiler/fheco_project /home/aimad/Desktop/projects/fhe_compiler/fheco_project /home/aimad/Desktop/projects/fhe_compiler/fheco_project /home/aimad/Desktop/projects/fhe_compiler/fheco_project /home/aimad/Desktop/projects/fhe_compiler/fheco_project/CMakeFiles/dsl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dsl.dir/depend

