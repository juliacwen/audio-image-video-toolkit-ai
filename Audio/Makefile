CXX = g++
CXXFLAGS = -std=c++17 -O2 -I/opt/homebrew/include -Icpp/inc 
LDFLAGS = -L/opt/homebrew/lib -lsndfile -lgtest -lgtest_main -pthread -lm 

# Binaries
TOOLS = wav_to_csv wav_freq_csv wav_freq_csv_channelized wav_comp_diff
TESTS = test_wav_to_csv test_wav_freq_csv test_wav_freq_csv_multi_channel

all: $(TOOLS) $(TESTS)

# Tools
wav_to_csv: cpp/src/wav_to_csv.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

wav_freq_csv: cpp/src/wav_freq_csv.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

wav_freq_csv_channelized: cpp/src/wav_freq_csv_channelized.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

wav_comp_diff: cpp/src/wav_comp_diff.cpp 
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


# Tests
test_wav_to_csv: cpp/tests/test_wav_to_csv.cpp wav_to_csv
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_wav_freq_csv: cpp/tests/test_wav_freq_csv.cpp wav_freq_csv
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_wav_freq_csv_multi_channel: cpp/tests/test_wav_freq_csv_multi_channel.cpp wav_freq_csv_channelized
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Run all tests
test: $(TESTS)
	./test_wav_to_csv
	./test_wav_freq_csv
	./wav_freq_csv_multi_channel

clean:
	rm -f $(TOOLS) $(TESTS) *_ch*.csv
