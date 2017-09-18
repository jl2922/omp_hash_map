# Default options.
CXX := g++
CXXFLAGS := -std=c++14 -Wall -Wextra -O3 -fopenmp -g
SRC_DIR := src
OBJ_DIR := build
TEST_EXE := test.out

# Testsources and intermediate objects.
TESTS := $(shell find $(SRC_DIR) -name "*_test.cc")
HEADERS := $(shell find $(SRC_DIR) -name "*.h")
TEST_OBJS := $(TESTS:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

# GTest related.
GTEST_DIR := gtest-1.8.0/googletest
GTEST_CXXFLAGS := $(CXXFLAGS) -isystem $(GTEST_DIR)/include -pthread
GTEST_HEADERS := $(GTEST_DIR)/include/gtest/*.h \
		$(GTEST_DIR)/include/gtest/internal/*.h
GTEST_SRCS := $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h \
		$(GTEST_HEADERS)
GTEST_MAIN := $(OBJ_DIR)/gtest_main.a

# Host specific configurations.
ifeq ($(HOSTNAME), travis)
ifneq ($(TRAVIS_CXX),)
CXX := $(TRAVIS_CXX)
endif
endif

.PHONY: all test all_tests clean

all: test

test: $(TEST_EXE)
	./$(TEST_EXE) --gtest_filter=-*LargeTest.*

all_tests: $(TEST_EXE)
	./$(TEST_EXE)

clean:
	rm -rf $(OBJ_DIR)
	rm -f ./$(TEST_EXE)
	
# Tests.

$(TEST_EXE): $(TEST_OBJS) $(OBJS) $(GTEST_MAIN) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(OBJS) $(GTEST_MAIN) \
			-o $(TEST_EXE) $(LDLIBS) -lpthread

$(TEST_OBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(HEADERS)
	mkdir -p $(@D) && $(CXX) $(GTEST_CXXFLAGS) -c $< -o $@

$(GTEST_MAIN): $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest-all.o: $(GTEST_SRCS)
	mkdir -p $(@D) && $(CXX) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
			$(GTEST_DIR)/src/gtest-all.cc -o $@

$(OBJ_DIR)/gtest_main.o: $(GTEST_SRCS)
	mkdir -p $(@D) && $(CXX) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
			$(GTEST_DIR)/src/gtest_main.cc -o $@
