// include libraries
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <sstream>

// define the BinomialResult type
struct BinomialResult {
    double price;
    double delta;
    double bond_position;
    double risk_neutral_prob;
    bool is_arbitrage;
    double replication_error;
};

class OnePeriodBinomialPricer {
private:
    double S0, K, u, d, r;
    const double EPS = 1e-10;

    void validate_positive(double value, const std::string& name) {
        if (value <= 0) {
            throw std::invalid_argument(name + " must be positive");
        };
    };

    void validate_no_arbitrage() {
        double exp_r = std::exp(r);
        if (d >= exp_r - EPS || u <= exp_r + EPS) {
            std::cerr << "Warning: Arbitrage condition may be violated\n";
        };
    };

public:

    OnePeriodBinomialPricer(double S0_, double K_, double u_, double d_, double r_) : S0(S0_), K(K_), u(u_), d(d_), r(r_) {
        validate_positive(S0, "S0");
        validate_positive(K, "K");
        validate_positive(u, "u");
        validate_positive(d, "d");
        if (r <= -1.0) throw std::invalid_argument("r must be > -1");
        validate_no_arbitrage();
    };

    BinomialResult price_call() const {
        double Su = S0 * u;
        double Sd = S0 * d;
        double Cu = std::max(Su - K, 0.0);
        double Cd = std::max(Sd - K, 0.0);

        double q = (std::exp(r) - d) / (u - d);
        double discount = std::exp(-r);
        double price = discount * (q * Cu + (1.0 - q) * Cd);
        double delta = (Cu - Cd) / (Su - Sd);
        double bond = (Su * Cd - Sd * Cu) / ((Su - Sd) * std::exp(r));

        // replication error analysis
        double portfolio_up = delta * Su + bond * std::exp(r);
        double portfolio_down = delta * Sd + bond * std::exp(r);
        double replication_error = std::max(std::abs(portfolio_up - Cu), std::abs(portfolio_down - Cd));

        return {
            price, delta, bond, q,
            !(q > 0 && q < 1),
            replication_error
        };
    };

    std::string to_json(const BinomialResult& result) const {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"Price\": " << result.price << ",\n";
        oss << "  \"delta\": " << result.delta << ",\n";
        oss << "  \"bond_position\": " << result.bond_position << ",\n";
        oss << "  \"risk_neutral_prob\": " << result.risk_neutral_prob << ",\n";
        oss << "  \"is_arbitrage\": " << (result.is_arbitrage ? "true" : "false") << ",\n";
        oss << "  \"replication_error\": " << result.replication_error << "\n"; 
        oss << "}";
        return oss.str();
    };
};

int main() {
    try {
        OnePeriodBinomialPricer pricer(100.0, 100.0, 1.12, 0.92, 0.02);
        auto result = pricer.price_call();
        std::cout << pricer.to_json(result) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
};