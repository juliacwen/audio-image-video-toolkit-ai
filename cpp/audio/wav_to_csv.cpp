//=============================================================================
//  FileName:      wav_to_csv.cpp
//  Copyright (c)  2025  Julia Wen. All Rights Reserved.
//  Date:          August 18, 2025
//  Description:   wav_to_csv.cpp (input.wav output.csv [max_samples])
//=============================================================================

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

static uint16_t readU16(std::istream& in){ uint8_t b[2]; in.read((char*)b,2); return b[0] | (b[1]<<8); }
static uint32_t readU32(std::istream& in){ uint8_t b[4]; in.read((char*)b,4); return uint32_t(b[0]) | (uint32_t(b[1])<<8) | (uint32_t(b[2])<<16) | (uint32_t(b[3])<<24); }
static int16_t readI16(const uint8_t* p){ return int16_t(p[0] | (p[1]<<8)); }

// read signed 24-bit little-endian → int32_t
static int32_t readI24(const uint8_t* p){
    int32_t v = (p[0] | (p[1]<<8) | (p[2]<<16));
    if(v & 0x800000) v |= ~0xFFFFFF; // sign extend
    return v;
}

int main(int argc, char** argv){
    if(argc<3){ std::cerr<<"Usage: "<<argv[0]<<" input.wav output.csv [max_samples]\n"; return 1; }
    std::string inPath=argv[1], outPath=argv[2];
    size_t max_samples = (argc>=4)? std::stoul(argv[3]) : 0;

    std::ifstream f(inPath, std::ios::binary);
    if(!f){ std::cerr<<"Open fail: "<<inPath<<"\n"; return 1; }

    char riff[4]; f.read(riff,4);
    if(std::string(riff,4)!="RIFF"){ std::cerr<<"Not RIFF\n"; return 1; }
    readU32(f); // file size
    char wave[4]; f.read(wave,4);
    if(std::string(wave,4)!="WAVE"){ std::cerr<<"Not WAVE\n"; return 1; }

    bool have_fmt=false, have_data=false;
    uint16_t fmt=0, ch=0, bps=0; uint32_t sr=0;
    std::streampos dataPos{}; uint32_t dataSize=0;

    while(f && !(have_fmt && have_data)){
        char id[4]; if(!f.read(id,4)) break;
        uint32_t sz = readU32(f);
        std::string sid(id,4);

        if(sid=="fmt "){
            have_fmt=true;
            uint16_t audioFormat = readU16(f);     // PCM=1, Float=3
            ch  = readU16(f);                      // channels
            sr  = readU32(f);                      // sample rate
            readU32(f);                            // byte rate (skip)
            readU16(f);                            // block align (skip)
            bps = readU16(f);                      // bits per sample
            if(sz>16) f.seekg(sz-16, std::ios::cur); // skip extra fields
            fmt = audioFormat;
        }
        else if(sid=="data"){
            have_data=true;
            dataPos = f.tellg();
            dataSize = sz;
            f.seekg(sz+(sz&1), std::ios::cur); // skip
        }
        else {
            f.seekg(sz+(sz&1), std::ios::cur); // skip unknown chunks
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

    for(size_t i=0;i<N;++i){
        double sample=0.0;
        if(fmt==1 && bps==16){ // PCM16
            if(ch==1){
                sample = readI16(&raw[i*2]);
            } else {
                int32_t L=readI16(&raw[(i*ch+0)*2]);
                int32_t R=readI16(&raw[(i*ch+1)*2]);
                sample = (L+R)/2.0;
            }
        }
        else if(fmt==1 && bps==24){ // PCM24
            if(ch==1){
                sample = readI24(&raw[i*3]);
            } else {
                int32_t L=readI24(&raw[(i*ch+0)*3]);
                int32_t R=readI24(&raw[(i*ch+1)*3]);
                sample = (L+R)/2.0;
            }
        }
        else if(fmt==3 && bps==32){ // Float32
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
    }

    std::cout<<"Wrote "<<N<<" samples to "<<outPath<<" (sr="<<sr<<", ch="<<ch<<", bps="<<bps<<", fmt="<<fmt<<")\n";
    return 0;
}
