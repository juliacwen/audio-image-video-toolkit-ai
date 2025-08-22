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

static uint16_t RU16(std::istream& in, bool big){uint8_t b[2];in.read((char*)b,2);return big?(b[0]<<8|b[1]):(b[1]<<8|b[0]);}
static uint32_t RU32(std::istream& in, bool big){uint8_t b[4];in.read((char*)b,4);return big?(uint32_t(b[0])<<24|uint32_t(b[1])<<16|uint32_t(b[2])<<8|b[3]):(uint32_t(b[3])<<24|uint32_t(b[2])<<16|uint32_t(b[1])<<8|b[0]);}
static int16_t I16(const uint8_t* p, bool big){return big?int16_t(int16_t(p[0])<<8|p[1]):int16_t(int16_t(p[1])<<8|p[0]);}

int main(int argc, char** argv){
    if(argc<3){std::cerr<<"Usage: "<<argv[0]<<" input.wav output.csv [max_samples]\n";return 1;}
    std::string inPath=argv[1], outPath=argv[2];
    size_t max_samples = (argc>=4)? std::stoul(argv[3]) : 0;

    std::ifstream f(inPath, std::ios::binary); if(!f){std::cerr<<"Open fail: "<<inPath<<"\n";return 1;}
    char riff[4]; f.read(riff,4); bool big=false;
    if(std::string(riff,4)=="RIFF") big=false;
    else if(std::string(riff,4)=="RIFX") big=true;
    else {std::cerr<<"Not RIFF/RIFX\n"; return 1;}
    RU32(f,big); char wave[4]; f.read(wave,4); if(std::string(wave,4)!="WAVE"){std::cerr<<"Not WAVE\n";return 1;}

    bool have_fmt=false, have_data=false; uint16_t fmt=0,ch=0,bps=0; uint32_t sr=0; std::streampos dataPos{}; uint32_t dataSize=0;
    while(f && !(have_fmt&&have_data)){
        char id[4]; if(!f.read(id,4)) break; uint32_t sz=RU32(f,big); std::string sid(id,4);
        if(sid=="fmt "){
            have_fmt=true; std::vector<uint8_t> buf(sz); f.read((char*)buf.data(),sz);
            fmt = big?(buf[0]<<8|buf[1]):(buf[1]<<8|buf[0]);
            ch  = big?(buf[2]<<8|buf[3]):(buf[3]<<8|buf[2]);
            sr  = big?(uint32_t(buf[4])<<24|uint32_t(buf[5])<<16|uint32_t(buf[6])<<8|buf[7])
                      :(uint32_t(buf[7])<<24|uint32_t(buf[6])<<16|uint32_t(buf[5])<<8|buf[4]);
            bps = big?(buf[14]<<8|buf[15]):(buf[15]<<8|buf[14]);
        } else if(sid=="data"){
            have_data=true; dataPos=f.tellg(); dataSize=sz; f.seekg(sz+(sz&1), std::ios::cur);
        } else {
            f.seekg(sz+(sz&1), std::ios::cur);
        }
    }
    if(!have_fmt||!have_data){std::cerr<<"Missing fmt/data\n";return 1;}
    if(fmt!=1||bps!=16){std::cerr<<"Only 16-bit PCM supported\n";return 1;}

    f.clear(); f.seekg(dataPos);
    std::vector<uint8_t> raw(dataSize); if(!f.read((char*)raw.data(),dataSize)){std::cerr<<"Read data fail\n";return 1;}
    size_t frames = dataSize/(2*ch);

    std::ofstream csv(outPath); if(!csv){std::cerr<<"Open output fail: "<<outPath<<"\n";return 1;}
    csv<<"Index,Sample\n";
    size_t N = (max_samples==0)? frames : std::min(frames, max_samples);
    for(size_t i=0;i<N;++i){
        if(ch==1){ csv<<i<<","<<I16(&raw[i*2],big)<<"\n"; }
        else {
            int32_t L=I16(&raw[(i*ch+0)*2],big), R=I16(&raw[(i*ch+1)*2],big);
            csv<<i<<","<<int16_t((L+R)/2)<<"\n";
        }
    }
    std::cout<<"Wrote "<<N<<" samples to "<<outPath<<" (sr="<<sr<<")\n";
    return 0;
}