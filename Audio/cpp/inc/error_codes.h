//=========================== error_codes.h ===========================

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

// Success
constexpr int SUCCESS = 0;

// WAV parsing errors
constexpr int ERR_FILE_NOT_FOUND      = -1;
constexpr int ERR_INVALID_HEADER      = -2;
constexpr int ERR_UNSUPPORTED_FORMAT  = -3;
constexpr int ERR_CHANNEL_COUNT       = -4;
constexpr int ERR_SAMPLE_DEPTH        = -5;
constexpr int ERR_READ_FAILURE        = -6;

// FFT or processing errors
constexpr int ERR_FFT_SIZE_MISMATCH   = -10;
constexpr int ERR_FFT_MEMORY          = -11;
constexpr int ERR_FFT_COMPUTE         = -12;

// CSV output errors
constexpr int ERR_CSV_WRITE_FAILURE   = -20;
constexpr int ERR_OUTPUT_PATH         = -21;

// Threading or system errors
constexpr int ERR_THREAD_LAUNCH       = -30;
constexpr int ERR_MUTEX_LOCK          = -31;
constexpr int ERR_PARALLEL_PROCESS    = -32;

// General unexpected failure
constexpr int ERR_INVALID_INPUT       = -98;
constexpr int ERR_UNKNOWN             = -99;

#endif // ERROR_CODES_H

