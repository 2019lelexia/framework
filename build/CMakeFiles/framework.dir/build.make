# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zml/desktop/framework

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zml/desktop/framework/build

# Include any dependencies generated for this target.
include CMakeFiles/framework.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/framework.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/framework.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/framework.dir/flags.make

CMakeFiles/framework.dir/main.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/main.cpp.o: /home/zml/desktop/framework/main.cpp
CMakeFiles/framework.dir/main.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/framework.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/main.cpp.o -MF CMakeFiles/framework.dir/main.cpp.o.d -o CMakeFiles/framework.dir/main.cpp.o -c /home/zml/desktop/framework/main.cpp

CMakeFiles/framework.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/main.cpp > CMakeFiles/framework.dir/main.cpp.i

CMakeFiles/framework.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/main.cpp -o CMakeFiles/framework.dir/main.cpp.s

CMakeFiles/framework.dir/frame.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/frame.cpp.o: /home/zml/desktop/framework/frame.cpp
CMakeFiles/framework.dir/frame.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/framework.dir/frame.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/frame.cpp.o -MF CMakeFiles/framework.dir/frame.cpp.o.d -o CMakeFiles/framework.dir/frame.cpp.o -c /home/zml/desktop/framework/frame.cpp

CMakeFiles/framework.dir/frame.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/frame.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/frame.cpp > CMakeFiles/framework.dir/frame.cpp.i

CMakeFiles/framework.dir/frame.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/frame.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/frame.cpp -o CMakeFiles/framework.dir/frame.cpp.s

CMakeFiles/framework.dir/global_params.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/global_params.cpp.o: /home/zml/desktop/framework/global_params.cpp
CMakeFiles/framework.dir/global_params.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/framework.dir/global_params.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/global_params.cpp.o -MF CMakeFiles/framework.dir/global_params.cpp.o.d -o CMakeFiles/framework.dir/global_params.cpp.o -c /home/zml/desktop/framework/global_params.cpp

CMakeFiles/framework.dir/global_params.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/global_params.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/global_params.cpp > CMakeFiles/framework.dir/global_params.cpp.i

CMakeFiles/framework.dir/global_params.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/global_params.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/global_params.cpp -o CMakeFiles/framework.dir/global_params.cpp.s

CMakeFiles/framework.dir/light_affine.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/light_affine.cpp.o: /home/zml/desktop/framework/light_affine.cpp
CMakeFiles/framework.dir/light_affine.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/framework.dir/light_affine.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/light_affine.cpp.o -MF CMakeFiles/framework.dir/light_affine.cpp.o.d -o CMakeFiles/framework.dir/light_affine.cpp.o -c /home/zml/desktop/framework/light_affine.cpp

CMakeFiles/framework.dir/light_affine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/light_affine.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/light_affine.cpp > CMakeFiles/framework.dir/light_affine.cpp.i

CMakeFiles/framework.dir/light_affine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/light_affine.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/light_affine.cpp -o CMakeFiles/framework.dir/light_affine.cpp.s

CMakeFiles/framework.dir/pixel_select.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/pixel_select.cpp.o: /home/zml/desktop/framework/pixel_select.cpp
CMakeFiles/framework.dir/pixel_select.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/framework.dir/pixel_select.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/pixel_select.cpp.o -MF CMakeFiles/framework.dir/pixel_select.cpp.o.d -o CMakeFiles/framework.dir/pixel_select.cpp.o -c /home/zml/desktop/framework/pixel_select.cpp

CMakeFiles/framework.dir/pixel_select.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/pixel_select.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/pixel_select.cpp > CMakeFiles/framework.dir/pixel_select.cpp.i

CMakeFiles/framework.dir/pixel_select.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/pixel_select.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/pixel_select.cpp -o CMakeFiles/framework.dir/pixel_select.cpp.s

CMakeFiles/framework.dir/point.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/point.cpp.o: /home/zml/desktop/framework/point.cpp
CMakeFiles/framework.dir/point.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/framework.dir/point.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/point.cpp.o -MF CMakeFiles/framework.dir/point.cpp.o.d -o CMakeFiles/framework.dir/point.cpp.o -c /home/zml/desktop/framework/point.cpp

CMakeFiles/framework.dir/point.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/point.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/point.cpp > CMakeFiles/framework.dir/point.cpp.i

CMakeFiles/framework.dir/point.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/point.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/point.cpp -o CMakeFiles/framework.dir/point.cpp.s

CMakeFiles/framework.dir/process_image.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/process_image.cpp.o: /home/zml/desktop/framework/process_image.cpp
CMakeFiles/framework.dir/process_image.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/framework.dir/process_image.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/process_image.cpp.o -MF CMakeFiles/framework.dir/process_image.cpp.o.d -o CMakeFiles/framework.dir/process_image.cpp.o -c /home/zml/desktop/framework/process_image.cpp

CMakeFiles/framework.dir/process_image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/process_image.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/process_image.cpp > CMakeFiles/framework.dir/process_image.cpp.i

CMakeFiles/framework.dir/process_image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/process_image.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/process_image.cpp -o CMakeFiles/framework.dir/process_image.cpp.s

CMakeFiles/framework.dir/tracker.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/tracker.cpp.o: /home/zml/desktop/framework/tracker.cpp
CMakeFiles/framework.dir/tracker.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/framework.dir/tracker.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/tracker.cpp.o -MF CMakeFiles/framework.dir/tracker.cpp.o.d -o CMakeFiles/framework.dir/tracker.cpp.o -c /home/zml/desktop/framework/tracker.cpp

CMakeFiles/framework.dir/tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/tracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/tracker.cpp > CMakeFiles/framework.dir/tracker.cpp.i

CMakeFiles/framework.dir/tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/tracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/tracker.cpp -o CMakeFiles/framework.dir/tracker.cpp.s

CMakeFiles/framework.dir/tracker2ml.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/tracker2ml.cpp.o: /home/zml/desktop/framework/tracker2ml.cpp
CMakeFiles/framework.dir/tracker2ml.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/framework.dir/tracker2ml.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/tracker2ml.cpp.o -MF CMakeFiles/framework.dir/tracker2ml.cpp.o.d -o CMakeFiles/framework.dir/tracker2ml.cpp.o -c /home/zml/desktop/framework/tracker2ml.cpp

CMakeFiles/framework.dir/tracker2ml.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/tracker2ml.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/tracker2ml.cpp > CMakeFiles/framework.dir/tracker2ml.cpp.i

CMakeFiles/framework.dir/tracker2ml.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/tracker2ml.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/tracker2ml.cpp -o CMakeFiles/framework.dir/tracker2ml.cpp.s

CMakeFiles/framework.dir/trajectory.cpp.o: CMakeFiles/framework.dir/flags.make
CMakeFiles/framework.dir/trajectory.cpp.o: /home/zml/desktop/framework/trajectory.cpp
CMakeFiles/framework.dir/trajectory.cpp.o: CMakeFiles/framework.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/framework.dir/trajectory.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/framework.dir/trajectory.cpp.o -MF CMakeFiles/framework.dir/trajectory.cpp.o.d -o CMakeFiles/framework.dir/trajectory.cpp.o -c /home/zml/desktop/framework/trajectory.cpp

CMakeFiles/framework.dir/trajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/framework.dir/trajectory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/trajectory.cpp > CMakeFiles/framework.dir/trajectory.cpp.i

CMakeFiles/framework.dir/trajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/framework.dir/trajectory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/trajectory.cpp -o CMakeFiles/framework.dir/trajectory.cpp.s

# Object files for target framework
framework_OBJECTS = \
"CMakeFiles/framework.dir/main.cpp.o" \
"CMakeFiles/framework.dir/frame.cpp.o" \
"CMakeFiles/framework.dir/global_params.cpp.o" \
"CMakeFiles/framework.dir/light_affine.cpp.o" \
"CMakeFiles/framework.dir/pixel_select.cpp.o" \
"CMakeFiles/framework.dir/point.cpp.o" \
"CMakeFiles/framework.dir/process_image.cpp.o" \
"CMakeFiles/framework.dir/tracker.cpp.o" \
"CMakeFiles/framework.dir/tracker2ml.cpp.o" \
"CMakeFiles/framework.dir/trajectory.cpp.o"

# External object files for target framework
framework_EXTERNAL_OBJECTS =

framework: CMakeFiles/framework.dir/main.cpp.o
framework: CMakeFiles/framework.dir/frame.cpp.o
framework: CMakeFiles/framework.dir/global_params.cpp.o
framework: CMakeFiles/framework.dir/light_affine.cpp.o
framework: CMakeFiles/framework.dir/pixel_select.cpp.o
framework: CMakeFiles/framework.dir/point.cpp.o
framework: CMakeFiles/framework.dir/process_image.cpp.o
framework: CMakeFiles/framework.dir/tracker.cpp.o
framework: CMakeFiles/framework.dir/tracker2ml.cpp.o
framework: CMakeFiles/framework.dir/trajectory.cpp.o
framework: CMakeFiles/framework.dir/build.make
framework: /usr/local/lib/libopencv_gapi.so.4.9.0
framework: /usr/local/lib/libopencv_stitching.so.4.9.0
framework: /usr/local/lib/libopencv_alphamat.so.4.9.0
framework: /usr/local/lib/libopencv_aruco.so.4.9.0
framework: /usr/local/lib/libopencv_bgsegm.so.4.9.0
framework: /usr/local/lib/libopencv_bioinspired.so.4.9.0
framework: /usr/local/lib/libopencv_ccalib.so.4.9.0
framework: /usr/local/lib/libopencv_dnn_objdetect.so.4.9.0
framework: /usr/local/lib/libopencv_dnn_superres.so.4.9.0
framework: /usr/local/lib/libopencv_dpm.so.4.9.0
framework: /usr/local/lib/libopencv_face.so.4.9.0
framework: /usr/local/lib/libopencv_freetype.so.4.9.0
framework: /usr/local/lib/libopencv_fuzzy.so.4.9.0
framework: /usr/local/lib/libopencv_hdf.so.4.9.0
framework: /usr/local/lib/libopencv_hfs.so.4.9.0
framework: /usr/local/lib/libopencv_img_hash.so.4.9.0
framework: /usr/local/lib/libopencv_intensity_transform.so.4.9.0
framework: /usr/local/lib/libopencv_line_descriptor.so.4.9.0
framework: /usr/local/lib/libopencv_mcc.so.4.9.0
framework: /usr/local/lib/libopencv_quality.so.4.9.0
framework: /usr/local/lib/libopencv_rapid.so.4.9.0
framework: /usr/local/lib/libopencv_reg.so.4.9.0
framework: /usr/local/lib/libopencv_rgbd.so.4.9.0
framework: /usr/local/lib/libopencv_saliency.so.4.9.0
framework: /usr/local/lib/libopencv_sfm.so.4.9.0
framework: /usr/local/lib/libopencv_stereo.so.4.9.0
framework: /usr/local/lib/libopencv_structured_light.so.4.9.0
framework: /usr/local/lib/libopencv_superres.so.4.9.0
framework: /usr/local/lib/libopencv_surface_matching.so.4.9.0
framework: /usr/local/lib/libopencv_tracking.so.4.9.0
framework: /usr/local/lib/libopencv_videostab.so.4.9.0
framework: /usr/local/lib/libopencv_viz.so.4.9.0
framework: /usr/local/lib/libopencv_wechat_qrcode.so.4.9.0
framework: /usr/local/lib/libopencv_xfeatures2d.so.4.9.0
framework: /usr/local/lib/libopencv_xobjdetect.so.4.9.0
framework: /usr/local/lib/libopencv_xphoto.so.4.9.0
framework: /usr/local/lib/libfmt.a
framework: /usr/local/lib/libpcl_surface.so
framework: /usr/local/lib/libpcl_keypoints.so
framework: /usr/local/lib/libpcl_tracking.so
framework: /usr/local/lib/libpcl_recognition.so
framework: /usr/local/lib/libpcl_stereo.so
framework: /usr/local/lib/libpcl_outofcore.so
framework: /usr/local/lib/libpcl_people.so
framework: /usr/lib/libOpenNI.so
framework: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
framework: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
framework: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
framework: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
framework: /usr/local/lib/libopencv_shape.so.4.9.0
framework: /usr/local/lib/libopencv_highgui.so.4.9.0
framework: /usr/local/lib/libopencv_datasets.so.4.9.0
framework: /usr/local/lib/libopencv_plot.so.4.9.0
framework: /usr/local/lib/libopencv_text.so.4.9.0
framework: /usr/local/lib/libopencv_ml.so.4.9.0
framework: /usr/local/lib/libopencv_phase_unwrapping.so.4.9.0
framework: /usr/local/lib/libopencv_optflow.so.4.9.0
framework: /usr/local/lib/libopencv_ximgproc.so.4.9.0
framework: /usr/local/lib/libopencv_video.so.4.9.0
framework: /usr/local/lib/libopencv_videoio.so.4.9.0
framework: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
framework: /usr/local/lib/libopencv_objdetect.so.4.9.0
framework: /usr/local/lib/libopencv_calib3d.so.4.9.0
framework: /usr/local/lib/libopencv_dnn.so.4.9.0
framework: /usr/local/lib/libopencv_features2d.so.4.9.0
framework: /usr/local/lib/libopencv_flann.so.4.9.0
framework: /usr/local/lib/libopencv_photo.so.4.9.0
framework: /usr/local/lib/libopencv_imgproc.so.4.9.0
framework: /usr/local/lib/libopencv_core.so.4.9.0
framework: /usr/local/lib/libpcl_registration.so
framework: /usr/local/lib/libpcl_segmentation.so
framework: /usr/local/lib/libpcl_features.so
framework: /usr/local/lib/libpcl_filters.so
framework: /usr/local/lib/libpcl_sample_consensus.so
framework: /usr/local/lib/libpcl_ml.so
framework: /usr/local/lib/libpcl_visualization.so
framework: /usr/local/lib/libpcl_search.so
framework: /usr/local/lib/libpcl_kdtree.so
framework: /usr/local/lib/libpcl_io.so
framework: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
framework: /usr/lib/x86_64-linux-gnu/libpthread.a
framework: /usr/local/lib/libpcl_octree.so
framework: /usr/lib/x86_64-linux-gnu/libpng.so
framework: /usr/lib/x86_64-linux-gnu/libz.so
framework: /usr/lib/libOpenNI.so
framework: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
framework: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
framework: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
framework: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libfreetype.so
framework: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libGLEW.so
framework: /usr/lib/x86_64-linux-gnu/libX11.so
framework: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
framework: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
framework: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
framework: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
framework: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
framework: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
framework: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
framework: /usr/local/lib/libpcl_common.so
framework: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
framework: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
framework: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
framework: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
framework: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
framework: CMakeFiles/framework.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable framework"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/framework.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/framework.dir/build: framework
.PHONY : CMakeFiles/framework.dir/build

CMakeFiles/framework.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/framework.dir/cmake_clean.cmake
.PHONY : CMakeFiles/framework.dir/clean

CMakeFiles/framework.dir/depend:
	cd /home/zml/desktop/framework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zml/desktop/framework /home/zml/desktop/framework /home/zml/desktop/framework/build /home/zml/desktop/framework/build /home/zml/desktop/framework/build/CMakeFiles/framework.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/framework.dir/depend
