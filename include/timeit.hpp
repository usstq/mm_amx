#pragma once

#include <chrono>
#include <thread>
#include <iostream>

uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}

struct RDTSC {
    uint64_t tsc_ticks_per_second;
    RDTSC() {
        tsc_ticks_per_second = rdtsc_calibrate();
        name = nullptr;
    }

    uint64_t st;
    const char * name;
    void start(const char * _name = nullptr) {
        if (name) {
            double dt = (__rdtsc() - st) * 1.0 / tsc_ticks_per_second;
            std::cout << " [RDTSC] : " << name << " took " << dt*1e6 << " us" << std::endl;
            name = nullptr;
        }
        name = _name;
        st = __rdtsc();
    }
    void end() {
        start(nullptr);
    }
};

uint64_t get_tsc_ticks_per_second() {
    static auto tsc_ticks_per_second = rdtsc_calibrate();
    return tsc_ticks_per_second;
}
double tsc2second(uint64_t diff) {
    return diff * 1.0/get_tsc_ticks_per_second();
}

uint64_t second2tsc(double sec) {
    return sec * get_tsc_ticks_per_second();
}

// timeit will record best latency for each problem in a csv log file
// and it will also show hint about whether it's improved or descreased
// over changes
struct timeit {
    const char * app_version;
    timeit() {
    }

    void set_app(const char * _app_version) {
        app_version = _app_version;
    }

    struct ANSIcolor {
        const char * code;
        ANSIcolor(const char * code = "0") : code(code){
        }
        friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
            out << "\033[" << obj.code << "m";
            return out;
        }
    };

    template<typename Callable>
    double operator()(
                      int expect_times_milliseconds,
                      const Callable & c,
                      double opsPerCall = 0,
                      double peakOpsPerSecond = 0,
                      const char * prob = nullptr) {
        int times;

        // cache warm-up
        c();
        c();

        // determine times
        if (expect_times_milliseconds > 0) {
            times = expect_times_milliseconds;
        } else {
            double expect_duration = -expect_times_milliseconds * 0.001;
            // estimate how many times required to reach the duration
            auto start = __rdtsc();
            c();
            auto oneshot = __rdtsc() - start;
            times = second2tsc(expect_duration)/oneshot;
        }

        // profiling
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < times; i++) {
            c();
        }
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> total_latency = finish-start;
        auto avg_latency = total_latency.count()/times;
        std::cout << ANSIcolor("0;33") << "Average latency : " << avg_latency*1e6 << " us x " << times;
        if (opsPerCall > 0 && peakOpsPerSecond > 0) {
            std::cout << "  HW Usage : " << static_cast<int>(100*(opsPerCall/avg_latency)/(peakOpsPerSecond)) << "% ("
                    << opsPerCall/avg_latency/(1e9) << " Gops /"
                    << peakOpsPerSecond/1e9 << " Gops)";
        }
        std::cout << ANSIcolor() << std::endl;
        return avg_latency;
    }
};


//=============================================================
// BF16-amx Peak (Gops)
// c += a*b is counted as 2 Ops
// 

constexpr double AMXBf16OpsPerTDP = (16*16*32)*2;
constexpr double AMXBf16TDPThrouput = 16;
constexpr double AMXBf16OpsPerCycleCore = AMXBf16OpsPerTDP/AMXBf16TDPThrouput;
constexpr double AMXBf16FreqGHz = 2.05;
constexpr double AMXBf16Freq2GHz = 3;//2.32;
constexpr double AMXBf16PeakGopsPerCore = AMXBf16OpsPerCycleCore * AMXBf16FreqGHz;
constexpr double AMXBf16PeakGops2PerCore = AMXBf16OpsPerCycleCore * AMXBf16Freq2GHz;

constexpr double AVX512FreqGHz = 3;//2.32;
constexpr double FP32OpsPerCycleCore = 64; // 2 AVX512_FMAs/cycle/core = 2*(16+16) Ops/cycle/core
constexpr double FP32PeakGopsPerCore = FP32OpsPerCycleCore * AVX512FreqGHz;
