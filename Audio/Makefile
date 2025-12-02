CXX = g++
CXXFLAGS = -std=c++17 -O2 -I/opt/homebrew/include -Icpp/inc 
LDFLAGS = -L/opt/homebrew/lib -lsndfile -lgtest -lgtest_main -pthread -lm 

# Library object files in obj directory
LIB_OBJECTS = obj/wav_utils.o obj/wav_writer.o obj/fft_utils.o

# Binaries
TOOLS = wav_to_csv wav_freq_csv wav_freq_csv_channelized wav_comp_diff
TESTS = test_wav_to_csv test_wav_freq_csv test_wav_freq_csv_multi_channel

all: $(TOOLS) $(TESTS)

# Create obj directory
obj:
	mkdir -p obj

# Compile library object files
obj/wav_utils.o: cpp/src/wav_utils.cpp | obj
	$(CXX) $(CXXFLAGS) -c cpp/src/wav_utils.cpp -o obj/wav_utils.o

obj/wav_writer.o: cpp/src/wav_writer.cpp | obj
	$(CXX) $(CXXFLAGS) -c cpp/src/wav_writer.cpp -o obj/wav_writer.o

obj/fft_utils.o: cpp/src/fft_utils.cpp | obj
	$(CXX) $(CXXFLAGS) -c cpp/src/fft_utils.cpp -o obj/fft_utils.o

# Tools
wav_to_csv: cpp/src/wav_to_csv.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

wav_freq_csv: cpp/src/wav_freq_csv.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

wav_freq_csv_channelized: cpp/src/wav_freq_csv_channelized.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm

wav_comp_diff: cpp/src/wav_comp_diff.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Tests
test_wav_to_csv: cpp/tests/test_wav_to_csv.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_wav_freq_csv: cpp/tests/test_wav_freq_csv.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test_wav_freq_csv_multi_channel: cpp/tests/test_wav_freq_csv_multi_channel.cpp $(LIB_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Run all tests
test: $(TESTS)
	./test_wav_to_csv
	./test_wav_freq_csv
	./test_wav_freq_csv_multi_channel

clean:
	rm -f $(TOOLS) $(TESTS) *_ch*.csv
	rm -rf obj

.PHONY: all test clean