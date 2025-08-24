CXX = g++
CXXFLAGS = -std=c++17 -O2 -I/opt/homebrew/include
LDFLAGS = -L/opt/homebrew/lib -lgtest -lgtest_main -pthread -lm

# Binaries
TOOLS = wav_to_csv wav_freq_csv
TESTS = test_wav_to_csv test_wav_freq_csv

all: $(TOOLS) $(TESTS)

# Tools
wav_to_csv: cpp/audio/wav_to_csv.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

wav_freq_csv: cpp/audio/wav_freq_csv.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

# Tests
test_wav_to_csv: cpp/tests/test_wav_to_csv.cpp wav_to_csv
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_wav_freq_csv: cpp/tests/test_wav_freq_csv.cpp wav_freq_csv
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Run all tests
test: $(TESTS)
	./test_wav_to_csv
	./test_wav_freq_csv

clean:
	rm -f $(TOOLS) $(TESTS) sine.wav sine.csv sine_spectrum.csv
