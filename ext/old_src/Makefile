
OUTPUT_FILENAME=run
TEST_FILENAME=test

MAIN_OBJECT:=build/src/main.o

S_DIR=src
B_DIR=build

EXEC=$(B_DIR)/$(OUTPUT_FILENAME)
TEST=$(B_DIR)/$(TEST_FILENAME)

EXT=cpp
EXT_CU=cu
CPP=nvcc
CPP_OPTS=-g -I$(S_DIR) -O3 
#$(shell pkg-config --cflags --libs /usr/local/Cellar/opencv@3/3.4.5_2/lib/pkgconfig/opencv.pc)

# Finds all of the sources with extension cpp
CU_SOURCES := $(wildcard src/*.$(EXT_CU))
SOURCES := $(wildcard src/*.$(EXT))
TEST_SOURCES := $(wildcard src/test/*.$(EXT))

# Makes a list of objects with the build directory appended to each source with source directory included
OBJECTS := $(addprefix $(B_DIR)/,$(SOURCES:.$(EXT)=.o))
CU_OBJECTS := $(addprefix $(B_DIR)/,$(CU_SOURCES:.$(EXT_CU)=.o))
TEST_OBJECTS := $(addprefix $(B_DIR)/,$(TEST_SOURCES:.$(EXT)=.o))

# Utility target that prints the variable value
print-%  : 
	@echo $* = $($*)
	# @echo $* = $(filter-out $(TEST_OBJECT), $(OBJECTS))

all: BuildMain


# Objects depends on each object depends on each cpp
$(OBJECTS): $(B_DIR)/%.o: %.$(EXT)
	# Make the subdirectory to retain src structure
	mkdir -p $(@D)
	# Build source file
	$(CPP) -c -o $@ $< $(CPP_OPTS)

$(CU_OBJECTS): $(B_DIR)/%.o: %.$(EXT_CU)
	# Make the subdirectory to retain src structure
	mkdir -p $(@D)
	# Build source file
	$(CPP) -c -o $@ $< $(CPP_OPTS)

# Build requires all objects
BuildMain: $(filter-out $(TEST_OBJECTS), $(OBJECTS)) $(CU_OBJECTS)
	# Link all objects and output executable
	$(CPP) $(CPP_OPTS) -o $(EXEC) $^ 

BuildTest: $(filter-out $(MAIN_OBJECT), $(OBJECTS)) $(CU_OBJECTS)
	# Link all objects and output executable
	$(CPP) $(CPP_OPTS) -o $(EXEC) $^

# Requesting executable requires build
$(EXEC): BuildMain

$(TEST): BuildTest

# Run requires executable, runs executable
run: BuildMain
	$(EXEC)


test: BuildTest
	$(EXEC)

# Clean recursively removes build directory
clean:
	rm -rf $(B_DIR)/*

profile_c: $(EXEC)
	nvprof $(EXEC)

profile: $(EXEC)
	gprof $(EXEC)

