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
include CMakeFiles/debug2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/debug2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/debug2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/debug2.dir/flags.make

CMakeFiles/debug2.dir/debug3.cpp.o: CMakeFiles/debug2.dir/flags.make
CMakeFiles/debug2.dir/debug3.cpp.o: /home/zml/desktop/framework/debug3.cpp
CMakeFiles/debug2.dir/debug3.cpp.o: CMakeFiles/debug2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/debug2.dir/debug3.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug2.dir/debug3.cpp.o -MF CMakeFiles/debug2.dir/debug3.cpp.o.d -o CMakeFiles/debug2.dir/debug3.cpp.o -c /home/zml/desktop/framework/debug3.cpp

CMakeFiles/debug2.dir/debug3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/debug2.dir/debug3.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zml/desktop/framework/debug3.cpp > CMakeFiles/debug2.dir/debug3.cpp.i

CMakeFiles/debug2.dir/debug3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/debug2.dir/debug3.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zml/desktop/framework/debug3.cpp -o CMakeFiles/debug2.dir/debug3.cpp.s

# Object files for target debug2
debug2_OBJECTS = \
"CMakeFiles/debug2.dir/debug3.cpp.o"

# External object files for target debug2
debug2_EXTERNAL_OBJECTS =

debug2: CMakeFiles/debug2.dir/debug3.cpp.o
debug2: CMakeFiles/debug2.dir/build.make
debug2: /usr/local/lib/libopencv_gapi.so.4.9.0
debug2: /usr/local/lib/libopencv_stitching.so.4.9.0
debug2: /usr/local/lib/libopencv_alphamat.so.4.9.0
debug2: /usr/local/lib/libopencv_aruco.so.4.9.0
debug2: /usr/local/lib/libopencv_bgsegm.so.4.9.0
debug2: /usr/local/lib/libopencv_bioinspired.so.4.9.0
debug2: /usr/local/lib/libopencv_ccalib.so.4.9.0
debug2: /usr/local/lib/libopencv_dnn_objdetect.so.4.9.0
debug2: /usr/local/lib/libopencv_dnn_superres.so.4.9.0
debug2: /usr/local/lib/libopencv_dpm.so.4.9.0
debug2: /usr/local/lib/libopencv_face.so.4.9.0
debug2: /usr/local/lib/libopencv_freetype.so.4.9.0
debug2: /usr/local/lib/libopencv_fuzzy.so.4.9.0
debug2: /usr/local/lib/libopencv_hdf.so.4.9.0
debug2: /usr/local/lib/libopencv_hfs.so.4.9.0
debug2: /usr/local/lib/libopencv_img_hash.so.4.9.0
debug2: /usr/local/lib/libopencv_intensity_transform.so.4.9.0
debug2: /usr/local/lib/libopencv_line_descriptor.so.4.9.0
debug2: /usr/local/lib/libopencv_mcc.so.4.9.0
debug2: /usr/local/lib/libopencv_quality.so.4.9.0
debug2: /usr/local/lib/libopencv_rapid.so.4.9.0
debug2: /usr/local/lib/libopencv_reg.so.4.9.0
debug2: /usr/local/lib/libopencv_rgbd.so.4.9.0
debug2: /usr/local/lib/libopencv_saliency.so.4.9.0
debug2: /usr/local/lib/libopencv_sfm.so.4.9.0
debug2: /usr/local/lib/libopencv_stereo.so.4.9.0
debug2: /usr/local/lib/libopencv_structured_light.so.4.9.0
debug2: /usr/local/lib/libopencv_superres.so.4.9.0
debug2: /usr/local/lib/libopencv_surface_matching.so.4.9.0
debug2: /usr/local/lib/libopencv_tracking.so.4.9.0
debug2: /usr/local/lib/libopencv_videostab.so.4.9.0
debug2: /usr/local/lib/libopencv_viz.so.4.9.0
debug2: /usr/local/lib/libopencv_wechat_qrcode.so.4.9.0
debug2: /usr/local/lib/libopencv_xfeatures2d.so.4.9.0
debug2: /usr/local/lib/libopencv_xobjdetect.so.4.9.0
debug2: /usr/local/lib/libopencv_xphoto.so.4.9.0
debug2: /usr/local/lib/libfmt.a
debug2: /usr/local/lib/libpcl_surface.so
debug2: /usr/local/lib/libpcl_keypoints.so
debug2: /usr/local/lib/libpcl_tracking.so
debug2: /usr/local/lib/libpcl_recognition.so
debug2: /usr/local/lib/libpcl_stereo.so
debug2: /usr/local/lib/libpcl_outofcore.so
debug2: /usr/local/lib/libpcl_people.so
debug2: /usr/lib/libOpenNI.so
debug2: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
debug2: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
debug2: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
debug2: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
debug2: /usr/local/lib/libopencv_shape.so.4.9.0
debug2: /usr/local/lib/libopencv_highgui.so.4.9.0
debug2: /usr/local/lib/libopencv_datasets.so.4.9.0
debug2: /usr/local/lib/libopencv_plot.so.4.9.0
debug2: /usr/local/lib/libopencv_text.so.4.9.0
debug2: /usr/local/lib/libopencv_ml.so.4.9.0
debug2: /usr/local/lib/libopencv_phase_unwrapping.so.4.9.0
debug2: /usr/local/lib/libopencv_optflow.so.4.9.0
debug2: /usr/local/lib/libopencv_ximgproc.so.4.9.0
debug2: /usr/local/lib/libopencv_video.so.4.9.0
debug2: /usr/local/lib/libopencv_videoio.so.4.9.0
debug2: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
debug2: /usr/local/lib/libopencv_objdetect.so.4.9.0
debug2: /usr/local/lib/libopencv_calib3d.so.4.9.0
debug2: /usr/local/lib/libopencv_dnn.so.4.9.0
debug2: /usr/local/lib/libopencv_features2d.so.4.9.0
debug2: /usr/local/lib/libopencv_flann.so.4.9.0
debug2: /usr/local/lib/libopencv_photo.so.4.9.0
debug2: /usr/local/lib/libopencv_imgproc.so.4.9.0
debug2: /usr/local/lib/libopencv_core.so.4.9.0
debug2: /usr/local/lib/libpcl_registration.so
debug2: /usr/local/lib/libpcl_segmentation.so
debug2: /usr/local/lib/libpcl_features.so
debug2: /usr/local/lib/libpcl_filters.so
debug2: /usr/local/lib/libpcl_sample_consensus.so
debug2: /usr/local/lib/libpcl_ml.so
debug2: /usr/local/lib/libpcl_visualization.so
debug2: /usr/local/lib/libpcl_search.so
debug2: /usr/local/lib/libpcl_kdtree.so
debug2: /usr/local/lib/libpcl_io.so
debug2: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
debug2: /usr/lib/x86_64-linux-gnu/libpthread.a
debug2: /usr/local/lib/libpcl_octree.so
debug2: /usr/lib/x86_64-linux-gnu/libpng.so
debug2: /usr/lib/x86_64-linux-gnu/libz.so
debug2: /usr/lib/libOpenNI.so
debug2: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
debug2: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
debug2: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
debug2: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libfreetype.so
debug2: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libGLEW.so
debug2: /usr/lib/x86_64-linux-gnu/libX11.so
debug2: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
debug2: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
debug2: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
debug2: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
debug2: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
debug2: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
debug2: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
debug2: /usr/local/lib/libpcl_common.so
debug2: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
debug2: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
debug2: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
debug2: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
debug2: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
debug2: CMakeFiles/debug2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/zml/desktop/framework/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable debug2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/debug2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/debug2.dir/build: debug2
.PHONY : CMakeFiles/debug2.dir/build

CMakeFiles/debug2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/debug2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/debug2.dir/clean

CMakeFiles/debug2.dir/depend:
	cd /home/zml/desktop/framework/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zml/desktop/framework /home/zml/desktop/framework /home/zml/desktop/framework/build /home/zml/desktop/framework/build /home/zml/desktop/framework/build/CMakeFiles/debug2.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/debug2.dir/depend

