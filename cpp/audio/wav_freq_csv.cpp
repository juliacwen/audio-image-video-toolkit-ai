//=============================================================================
//  FileName:      wav_to_csv.cpp
//  Extended:      Also compute FFT spectrum -> output_spectrum.csv
//=============================================================================

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <complex>

using cd = std::complex<double>;
const double PI = acos(-1);

// Bit-reversal permutation
void bitReverse(std::vector<cd>& a) {
    int n = a.size();
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// Iterative FFT
void fft(std::vector<cd>& a, bool invert) {
    int n = a.size();
    bitReverse(a);
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i + j];
                cd v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (cd& x : a) x /= n;
    }
}

static uint16_t RU16(std::istream& in){ uint8_t b[2]; in.read((char*)b,2); return b[0] | (b[1]<<8); }
static uint32_t RU32(std::istream& in){ uint8_t b[4]; in.read((char*)b,4); return uint32_t(b[0]) | (uint32_t(b[1])<<8) | (uint32_t(b[2])<<16) | (uint32_t(b[3])<<24); }
static int16_t I16(const uint8_t* p){ return int16_t(p[0] | (p[1]<<8)); }
static int32_t I24(const uint8_t* p){ int32_t v = (p[0] | (p[1]<<8) | (p[2]<<16)); if(v & 0x800000) v |= ~0xFFFFFF; return v; }

int main(int argc, char** argv){
    if(argc<3){ std::cerr<<"Usage: "<<argv[0]<<" input.wav output.csv [max_samples]\n"; return 1; }
    std::string inPath=argv[1], outPath=argv[2];
    size_t max_samples = (argc>=4)? std::stoul(argv[3]) : 0;

    std::ifstream f(inPath, std::ios::binary);
    if(!f){ std::cerr<<"Open fail: "<<inPath<<"\n"; return 1; }

    char riff[4]; f.read(riff,4);
    if(std::string(riff,4)!="RIFF"){ std::cerr<<"Not RIFF\n"; return 1; }
    RU32(f);
    char wave[4]; f.read(wave,4);
    if(std::string(wave,4)!="WAVE"){ std::cerr<<"Not WAVE\n"; return 1; }

    bool have_fmt=false, have_data=false;
    uint16_t fmt=0, ch=0, bps=0; uint32_t sr=0;
    std::streampos dataPos{}; uint32_t dataSize=0;

    while(f && !(have_fmt && have_data)){
        char id[4]; if(!f.read(id,4)) break;
        uint32_t sz = RU32(f);
        std::string sid(id,4);
        if(sid=="fmt "){
            have_fmt=true;
            uint16_t audioFormat = RU16(f);
            ch  = RU16(f);
            sr  = RU32(f);
            RU32(f); RU16(f);
            bps = RU16(f);
            if(sz>16) f.seekg(sz-16, std::ios::cur);
            fmt = audioFormat;
        }
        else if(sid=="data"){
            have_data=true;
            dataPos = f.tellg();
            dataSize = sz;
            f.seekg(sz+(sz&1), std::ios::cur);
        }
        else {
            f.seekg(sz+(sz&1), std::ios::cur);
        }
    }

    if(!have_fmt || !have_data){ std::cerr<<"Missing fmt/data\n"; return 1; }
    if(!((fmt==1 && (bps==16 || bps==24)) || (fmt==3 && bps==32))){
        std::cerr<<"Only PCM 16/24-bit or Float32 supported\n";
        return 1;
    }

    f.clear(); f.seekg(dataPos);
    std::vector<uint8_t> raw(dataSize);
    if(!f.read((char*)raw.data(),dataSize)){ std::cerr<<"Read data fail\n"; return 1; }

    size_t sample_bytes = bps/8;
    size_t frames = dataSize/(sample_bytes*ch);

    std::ofstream csv(outPath);
    if(!csv){ std::cerr<<"Open output fail: "<<outPath<<"\n"; return 1; }
    csv<<"Index,Sample\n";

    size_t N = (max_samples==0)? frames : std::min(frames, max_samples);
    std::vector<double> samples;
    samples.reserve(N);

    for(size_t i=0;i<N;++i){
        double sample=0.0;
        if(fmt==1 && bps==16){
            if(ch==1) sample = I16(&raw[i*2]);
            else {
                int32_t L=I16(&raw[(i*ch+0)*2]);
                int32_t R=I16(&raw[(i*ch+1)*2]);
                sample = (L+R)/2.0;
            }
        }
        else if(fmt==1 && bps==24){
            if(ch==1) sample = I24(&raw[i*3]);
            else {
                int32_t L=I24(&raw[(i*ch+0)*3]);
                int32_t R=I24(&raw[(i*ch+1)*3]);
                sample = (L+R)/2.0;
            }
        }
        else if(fmt==3 && bps==32){
            if(ch==1){
                float v = *reinterpret_cast<const float*>(&raw[i*4]);
                sample = v;
            } else {
                float L = *reinterpret_cast<const float*>(&raw[(i*ch+0)*4]);
                float R = *reinterpret_cast<const float*>(&raw[(i*ch+1)*4]);
                sample = (L+R)/2.0;
            }
        }
        csv<<i<<","<<sample<<"\n";
        samples.push_back(sample);
    }
    csv.close();

    // ==== FFT ====
    size_t fftN = 1;
    while(fftN < samples.size()) fftN <<= 1;
    std::vector<cd> fa(fftN);
    for(size_t i=0;i<samples.size();i++) fa[i] = samples[i];
    for(size_t i=samples.size(); i<fftN; i++) fa[i] = 0;

    fft(fa, false);

    // derive spectrum filename
    std::string specPath = outPath;
    size_t dot = specPath.find_last_of('.');
    if(dot != std::string::npos) {
        specPath.insert(dot, "_spectrum");
    } else {
        specPath += "_spectrum.csv";
    }

    std::ofstream spec(specPath);
    spec<<"Frequency(Hz),Magnitude\n";
    for(size_t i=0; i<fftN/2; i++){ // only half spectrum
        double freq = (double)i * sr / fftN;
        double mag = std::abs(fa[i]);
        spec<<freq<<","<<mag<<"\n";
    }
    spec.close();

    std::cout<<"Wrote "<<N<<" samples to "<<outPath
             <<" and "<<specPath<<" ("<<sr<<" Hz)\n";
    return 0;
}
