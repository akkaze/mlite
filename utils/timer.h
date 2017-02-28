#ifndef MLITE_UTILS_TIMER_H_
#define MLITE_UTILS_TIMER_H_
#ifdef _MSC_VER
#include <windows.h>
#else
#include <sys/time.h>
#endif
namespace mlite {
class Timer {
#ifdef _MSC_VER
public:
	Timer() {
		QueryPerformanceFrequency(&freq_);
	}
	void Start() {
		QueryPerformanceCounter((LARGE_INTEGER*)&start_time_);
	}
	double Elapse() const {
		LARGE_INTEGER  elapsed;
		QueryPerformanceCounter((LARGE_INTEGER*)&end_time_);
		elapsed.QuadPart = end_time_.QuadPart - start_time_.QuadPart;
		return elapsed.QuadPart / static_cast<double>(freq_.QuadPart);
	}
private:
	LARGE_INTEGER freq_;
	LARGE_INTEGER start_time_;
	LARGE_INTEGER end_time_;
#else
public:
	Timer() : ts_(0) {}

	void Start() {
		struct timeval tval;
		gettimeofday(&tval, NULL);
		ts_ = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
	}

	double Elapse() const {
		struct timeval tval;
		gettimeofday(&tval, NULL);
		double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);

		return static_cast<double>(end_time - ts_) / 1000000.0;
	}

private:
	double ts_;
#endif
};
}
#endif