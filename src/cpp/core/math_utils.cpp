#include "math_utils.hpp"

namespace quant {

double mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot compute mean of empty vector");
    }
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
}

double variance(const std::vector<double>& data, int ddof) {
    if (data.size() <= static_cast<size_t>(ddof)) {
        throw std::invalid_argument("Not enough data points for variance calculation");
    }

    double m = mean(data);
    double sum_sq = 0.0;
    for (const auto& x : data) {
        double diff = x - m;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<double>(data.size() - ddof);
}

double std_dev(const std::vector<double>& data, int ddof) {
    return std::sqrt(variance(data, ddof));
}

double norm_cdf(double x) {
    // Using the error function: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double norm_pdf(double x) {
    // PDF of standard normal: (1/sqrt(2*pi)) * exp(-x^2/2)
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

} // namespace quant
