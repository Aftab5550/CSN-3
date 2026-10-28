// bounds.cpp — analytical p-value bounds for mean closeness under G(N,E)
// Compile:  g++ -O3 -march=native -fopenmp -std=c++17 bounds.cpp -o bounds
// Usage:    ./bounds [--data DIR] [--x-file CSV] [--orders 1,2,3,4]
//                    [--out CSV] [--detail CSV] [--threads 16]

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace std;
namespace fs = std::filesystem;

// ============================================================
// Structures
// ============================================================

struct Network {
    int N = 0;
    vector<pair<int,int>> edges;   // normalised (u < v)
    vector<vector<int>>   adj;
};

struct Bound {
    string name;
    int    order;
    double p;
    double value;
    bool   has_value;
    string note;
};

struct BoundResult {
    double best;
    double U_X, U_Z, L_Z, deterministic_max;
    vector<double> moments_Z;      // indexed by order 0..max_order
    vector<Bound>  bounds;
};

// ============================================================
// Read a network (edge-list OR adjacency-list, int OR string labels)
// Faithful port of Python's read_network.
// ============================================================

Network read_network(const string& path) {
    Network g;
    ifstream in(path);
    if (!in.is_open()) { cerr << "Cannot open " << path << "\n"; return g; }

    vector<string> lines;
    { string L; while (getline(in, L)) lines.push_back(L); }
    if (lines.empty()) return g;

    int N_header = 0;
    {
        istringstream h(lines[0]);
        string tok;
        if (h >> tok) { try { N_header = stoi(tok); } catch (...) {} }
    }

    // detect int vs string tokens
    bool all_int = true;
    for (size_t i = 1; i < lines.size() && all_int; ++i) {
        istringstream iss(lines[i]);
        string tok;
        while (iss >> tok) {
            try { stoi(tok); } catch (...) { all_int = false; break; }
        }
    }

    unordered_set<uint64_t> es;
    int max_id = -1;
    unordered_map<string,int> ids;

    for (size_t i = 1; i < lines.size(); ++i) {
        istringstream iss(lines[i]);
        string tok;
        if (!(iss >> tok)) continue;

        int a;
        if (all_int) {
            try { a = stoi(tok); } catch (...) { continue; }
            max_id = max(max_id, a);
        } else {
            auto it = ids.find(tok);
            if (it == ids.end()) { a = (int)ids.size(); ids[tok] = a; }
            else a = it->second;
        }

        while (iss >> tok) {
            int b;
            if (all_int) {
                try { b = stoi(tok); } catch (...) { continue; }
                max_id = max(max_id, b);
            } else {
                auto it = ids.find(tok);
                if (it == ids.end()) { b = (int)ids.size(); ids[tok] = b; }
                else b = it->second;
            }
            if (a == b) continue;
            int lo = min(a, b), hi = max(a, b);
            es.insert(((uint64_t)(uint32_t)lo << 32) | (uint32_t)hi);
        }
    }

    g.N = all_int ? max(N_header, max_id + 1) : max(N_header, (int)ids.size());
    g.edges.reserve(es.size());
    for (uint64_t key : es)
        g.edges.push_back({(int)(key >> 32), (int)(key & 0xFFFFFFFFu)});
    g.adj.assign(g.N, {});
    for (auto& e : g.edges) {
        g.adj[e.first].push_back(e.second);
        g.adj[e.second].push_back(e.first);
    }
    return g;
}

// ============================================================
// Exact mean closeness, parallel over sources.
//   C = (1 / (N(N-1))) * sum_{i != j} 1/d_ij
// ============================================================

double observed_closeness(const Network& g, int num_threads) {
    if (g.N < 2 || g.edges.empty()) return 0.0;
    const int N = g.N;
    double total = 0.0;

    #pragma omp parallel num_threads(num_threads) reduction(+:total)
    {
        vector<int> dist(N, -1);
        vector<int> q(N);

        #pragma omp for schedule(dynamic, 16)
        for (int src = 0; src < N; ++src) {
            if (g.adj[src].empty()) continue;

            int head = 0, tail = 0;
            q[tail++] = src;
            dist[src] = 0;

            while (head < tail) {
                int u = q[head++];
                int nd = dist[u] + 1;
                for (int v : g.adj[u])
                    if (dist[v] == -1) { dist[v] = nd; q[tail++] = v; }
            }

            double s = 0.0;
            for (int i = 1; i < tail; ++i) s += 1.0 / dist[q[i]];
            total += s;

            for (int i = 0; i < tail; ++i) dist[q[i]] = -1;
        }
    }

    return total / ((double)N * (N - 1));
}

// ============================================================
// Original first-moment bound + Markov
// ============================================================

double closeness_upper_original(int N, int E) {
    if (N < 2 || E == 0) return 0.0;
    long long M = (long long)N * (N - 1) / 2;
    double a = 1.0, used = 0.0, tot = 0.0;

    for (int l = 1; l < N && used < N - 1; ++l) {
        long long denom = M - l + 1;
        if (denom <= 0) break;
        long long factor = (long long)(N - l) * (E - l + 1);
        if (factor <= 0) a = 0.0;
        else             a *= (double)factor / (double)denom;
        if (!isfinite(a)) a = (double)(N - 1) - used;

        double m = min(a, (double)(N - 1) - used);
        if (m < 1e-12) break;
        tot  += m / l;
        used += m;
    }
    return tot / (N - 1);
}

double markov(double U, double x) {
    if (x <= 0.0) return 1.0;
    if (U < 0.0)  U = 0.0;
    return min(1.0, U / x);
}

// ============================================================
// S_r = sum_l  (l^{-r} - (l+1)^{-r}) * alpha_l    (last term l=N-1: l^{-r} * alpha)
// returns S[1..max_order]; delta, max_alpha out-parameters.
// ============================================================

vector<double> nonedge_S(int N, int E, int max_order,
                         double& delta_out, double& max_alpha_out)
{
    vector<double> S(max_order + 1, 0.0);
    long long M = (long long)N * (N - 1) / 2;
    double delta     = M > 0 ? (double)E / (double)M : 0.0;
    double max_alpha = 1.0 - delta;
    delta_out = delta; max_alpha_out = max_alpha;

    if (N < 3 || E < 2 || max_alpha <= 0.0) return S;

    double lam   = delta;
    double alpha = 0.0;

    for (int l = 2; l < N; ++l) {
        if (l <= E && M - l + 1 > 0) {
            lam *= (double)(N - l) * (E - l + 1) / (double)(M - l + 1);
            if (!isfinite(lam)) lam = numeric_limits<double>::infinity();
        } else lam = 0.0;

        double eta = (lam > 0.0 && M > l && M > E)
                     ? lam * (double)(M - E) / (double)(M - l) : 0.0;

        if (alpha < max_alpha)
            alpha = isfinite(eta) ? min(max_alpha, alpha + eta) : max_alpha;

        // iterative inverse powers:  lf^{-s}, nf^{-s} for s = 1..max_order
        const double lf = (double)l;
        const double nf = lf + 1.0;
        const double inv_lf = 1.0 / lf, inv_nf = 1.0 / nf;
        double p_lf = inv_lf, p_nf = inv_nf;

        if (l < N - 1) {
            for (int s = 1; s <= max_order; ++s) {
                S[s] += (p_lf - p_nf) * alpha;
                p_lf *= inv_lf; p_nf *= inv_nf;
            }
        } else {
            for (int s = 1; s <= max_order; ++s) {
                S[s] += p_lf * alpha;
                p_lf *= inv_lf;
            }
        }

        if (l < N - 1 && (alpha >= max_alpha || l >= E)) {
            const double nf2 = (double)(l + 1);
            const double inv_nf2 = 1.0 / nf2;
            double p = inv_nf2;
            for (int s = 1; s <= max_order; ++s) {
                S[s] += alpha * p;
                p *= inv_nf2;
            }
            break;
        }
    }

    for (int s = 1; s <= max_order; ++s) {
        double lim = max_alpha / pow(2.0, s);
        if (S[s] < 0.0)  S[s] = 0.0;
        if (S[s] > lim)  S[s] = lim;
    }
    return S;
}

// ============================================================
// Paley-Zygmund lower bound on Y = X - delta
// ============================================================

double lower_nonedge_Y(int N, int E) {
    long long M = (long long)N * (N - 1) / 2;
    if (M == 0) return 0.0;
    double delta = (double)E / (double)M;
    double c     = (1.0 - delta) / 2.0;
    if (N < 3 || E < 2 || c <= 0.0 || M < 3) return 0.0;

    double p2 = (double)E / M
              * (double)(E - 1) / (M - 1)
              * (double)(M - E) / (M - 2);
    double EY = (N - 2) * p2;
    if (EY <= 0.0) return 0.0;

    double EY2 = EY;
    if (N >= 4 && E >= 4 && M >= 5) {
        double p4 = (double)E / M
                  * (double)(E - 1) / (M - 1)
                  * (double)(E - 2) / (M - 2)
                  * (double)(E - 3) / (M - 3)
                  * (double)(M - E) / (M - 4);
        EY2 += (double)(N - 2) * (N - 3) * p4;
    }
    if (EY2 <= 0.0) return 0.0;

    double pz = min(1.0, EY * EY / EY2);
    return min(c, 0.5 * pz);
}

// ============================================================
// Main analytical bound routine
// ============================================================

static void add_best(BoundResult& r, const string& why = "minimum over all bounds tried") {
    double best = 1.0;
    for (auto& b : r.bounds) best = min(best, b.p);
    r.best = best;
    r.bounds.push_back({"best", 0, best, best, true, why});
}

BoundResult bound_closeness_p(int N, int E, double x,
                              const vector<int>& orders, int max_order)
{
    BoundResult r;
    r.best = 1.0;

    long long M = (long long)N * (N - 1) / 2;
    if (M == 0) {
        double p = (x <= 0.0) ? 1.0 : 0.0;
        r.best = p; r.U_X = 0.0; r.U_Z = 0.0; r.L_Z = 0.0; r.deterministic_max = 0.0;
        r.bounds.push_back({"trivial_M_zero", 0, p, 0.0, true, "M = 0"});
        add_best(r); return r;
    }

    double delta = (double)E / (double)M;
    double c     = (1.0 - delta) / 2.0;
    double dmax  = delta + c;
    r.deterministic_max = dmax;

    auto finish = [&](double p, double ux, double uz, double lz,
                      const string& nm, const string& note) {
        r.best = p; r.U_X = ux; r.U_Z = uz; r.L_Z = lz;
        r.bounds.push_back({nm, 0, p, 0.0, true, note});
        add_best(r); return r;
    };

    if (x <= delta)
        return finish(1.0, dmax, 1.0, 0.0, "edge_floor",
                      "x <= deterministic edge contribution delta");
    if (x > dmax) {
        if (x <= dmax + 1e-12) x = dmax;
        else return finish(0.0, dmax, 1.0, 0.0, "deterministic_max", "x > deterministic maximum");
    }
    if (c <= 0.0)
        return finish(x > delta ? 0.0 : 1.0, delta, 0.0, 0.0,
                      "complete_or_trivial_c_zero", "c = 0");

    double z = (x - delta) / c;
    if (z <= 0.0)
        return finish(1.0, dmax, 1.0, 0.0, "invalid_z", "z <= 0");
    if (z > 1.0) {
        if (z <= 1.0 + 1e-12) z = 1.0;
        else return finish(0.0, dmax, 1.0, 0.0, "deterministic_max", "z > 1");
    }

    double delta_o, max_alpha;
    vector<double> S = nonedge_S(N, E, max_order, delta_o, max_alpha);
    if (max_alpha <= 0.0)
        return finish(z > 0.0 ? 0.0 : 1.0, delta, 0.0, 0.0,
                      "no_nonedge_capacity", "max_alpha <= 0");

    // moments m_r = 2^r * S_r / max_alpha
    r.moments_Z.assign(max_order + 1, 0.0);
    for (int s : orders) {
        double v = pow(2.0, s) * S[s] / max_alpha;
        r.moments_Z[s] = max(0.0, min(1.0, v));
    }
    r.U_Z = r.moments_Z[1];

    // Paley-Zygmund lower bound on E[Z]
    double LY  = lower_nonedge_Y(N, E);
    double L_Z = (c > 0.0) ? LY / c : 0.0;
    L_Z = max(0.0, min(L_Z, r.U_Z));
    L_Z = min(L_Z, 1.0);
    r.L_Z = L_Z;

    // Markov moment bounds
    for (int s : orders) {
        double m_s = r.moments_Z[s];
        double p   = min(1.0, m_s / pow(z, s));
        char note[96];
        snprintf(note, sizeof(note), "m_%d / z^%d, z=%.17g", s, s, z);
        r.bounds.push_back({"moment_markov", s, p, m_s, true, note});
    }

    // Cantelli
    if (z > r.U_Z) {
        double m2 = r.moments_Z[2];
        double V  = max(0.0, m2 - L_Z * L_Z);
        double p_c = (V <= 0.0) ? 0.0 : V / (V + (z - r.U_Z) * (z - r.U_Z));
        r.bounds.push_back({"cantelli_m2", 2, p_c, V, true, "Var(Z) <= m_2 - L_Z^2"});

        double Vmax = (L_Z <= 0.5 && 0.5 <= r.U_Z)
                      ? 0.25
                      : max(L_Z * (1.0 - L_Z), r.U_Z * (1.0 - r.U_Z));
        double p_r = (Vmax <= 0.0) ? 0.0 : Vmax / (Vmax + (z - r.U_Z) * (z - r.U_Z));
        r.bounds.push_back({"cantelli_range", 0, p_r, Vmax, true,
                            "Var(Z) <= mu(1-mu), mu in [L_Z, U_Z]"});
    } else {
        r.bounds.push_back({"cantelli_m2",    2, 1.0, 0.0, false, "not applicable: z <= U_Z"});
        r.bounds.push_back({"cantelli_range", 0, 1.0, 0.0, false, "not applicable: z <= U_Z"});
    }

    r.U_X = delta + c * r.U_Z;
    add_best(r);
    return r;
}

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    string data_dir   = "data";
    string x_file;
    string orders_str = "1,2,3,4";
    string out_file   = "bounds_summary.csv";
    string detail_file= "bounds_detail.csv";
    int    nthreads   = 16;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto next = [&]() -> string { return (i + 1 < argc) ? argv[++i] : string(); };
        if      (a == "--data")    data_dir    = next();
        else if (a == "--x-file")  x_file      = next();
        else if (a == "--orders")  orders_str  = next();
        else if (a == "--out")     out_file    = next();
        else if (a == "--detail")  detail_file = next();
        else if (a == "--threads") nthreads    = max(1, stoi(next()));
    }

    // parse orders
    vector<int> orders;
    {
        istringstream iss(orders_str);
        string tok;
        while (getline(iss, tok, ',')) {
            try { orders.push_back(stoi(tok)); } catch (...) {}
        }
        sort(orders.begin(), orders.end());
        orders.erase(unique(orders.begin(), orders.end()), orders.end());
        if (orders.empty()) orders = {1,2,3,4};
        if (find(orders.begin(), orders.end(), 1) == orders.end())
            orders.insert(orders.begin(), 1);
    }
    int max_order = *max_element(orders.begin(), orders.end());

    // optional precomputed x
    unordered_map<string,double> xs;
    if (!x_file.empty()) {
        ifstream xf(x_file);
        vector<string> lines;
        string L;
        while (getline(xf, L)) lines.push_back(L);
        if (!lines.empty()) {
            if (lines[0].find("closeness") != string::npos) {
                // header-aware
                vector<string> cols;
                { istringstream hs(lines[0]); string t; while (getline(hs, t, ',')) cols.push_back(t); }
                int lc = -1, cc = -1;
                for (size_t j = 0; j < cols.size(); ++j) {
                    if (cols[j] == "language")  lc = (int)j;
                    if (cols[j] == "closeness") cc = (int)j;
                }
                if (lc >= 0 && cc >= 0) {
                    for (size_t i = 1; i < lines.size(); ++i) {
                        vector<string> r;
                        { istringstream rs(lines[i]); string t; while (getline(rs, t, ',')) r.push_back(t); }
                        if ((int)r.size() > max(lc, cc)) {
                            try { xs[r[lc]] = stod(r[cc]); } catch (...) {}
                        }
                    }
                }
            } else {
                // simple language,closeness
                for (auto& line : lines) {
                    vector<string> r;
                    { istringstream rs(line); string t; while (getline(rs, t, ',')) r.push_back(t); }
                    if (r.size() >= 2) { try { xs[r[0]] = stod(r[1]); } catch (...) {} }
                }
            }
        }
    }

    auto make_parent = [](const string& p) {
        auto par = fs::path(p).parent_path();
        if (!par.empty()) fs::create_directories(par);
    };
    make_parent(out_file); make_parent(detail_file);

    vector<fs::path> files;
    for (auto& e : fs::directory_iterator(data_dir))
        if (e.path().filename().string().find("_dependency_network.txt") != string::npos)
            files.push_back(e.path());
    sort(files.begin(), files.end());

    ofstream fsum(out_file), fdet(detail_file);
    fsum << "language,N,E,mean_k,delta,x_obs,U_old,p_old_markov,U_new,p_new_best,deterministic_max\n";
    fdet << "language,N,E,x_obs,bound,order,p_bound,value,note\n";
    fsum << setprecision(17);
    fdet << setprecision(17);

    printf("%-14s%8s%9s%8s%10s%10s%10s%12s%10s%12s%10s\n",
           "language","N","E","<k>","delta","x_obs","U_old","p_old","U_new","p_new","det_max");

    for (auto& path : files) {
        string lang = path.filename().string();
        auto pos = lang.find("_dependency_network.txt");
        if (pos != string::npos) lang = lang.substr(0, pos);

        Network g = read_network(path.string());
        int N = g.N, E = (int)g.edges.size();

        double x;
        if (xs.count(lang)) x = xs[lang];
        else {
            int eff = (N < 500) ? 1 : nthreads;     // small graphs: OpenMP overhead dominates
            x = observed_closeness(g, eff);
        }
        if (!isfinite(x)) x = 0.0;

        double mean_k = (N > 1) ? 2.0 * E / N : 0.0;
        double delta  = (N > 1) ? 2.0 * E / ((double)N * (N - 1)) : 0.0;
        double dmax   = (N > 1) ? (1.0 + delta) / 2.0 : 0.0;

        double U_old = closeness_upper_original(N, E);
        U_old = min(max(0.0, U_old), dmax);
        double p_old = markov(U_old, x);

        BoundResult r = bound_closeness_p(N, E, x, orders, max_order);

        fsum << lang << ',' << N << ',' << E << ','
             << mean_k << ',' << delta << ',' << x << ','
             << U_old  << ',' << p_old << ','
             << r.U_X  << ',' << r.best << ',' << dmax << '\n';

        fdet << lang << ',' << N << ',' << E << ',' << x
             << ",old_markov,0," << p_old << ',' << U_old
             << ",original first-moment Markov bound\n";
        for (auto& b : r.bounds) {
            fdet << lang << ',' << N << ',' << E << ',' << x
                 << ',' << b.name << ',' << b.order << ',' << b.p << ',';
            if (b.has_value) fdet << b.value;
            fdet << ',' << b.note << '\n';
        }
        fsum.flush(); fdet.flush();

        printf("%-14s%8d%9d%8.3f%10.3e%10.6f%10.6f%12.4e%10.6f%12.4e%10.6f\n",
               lang.c_str(), N, E, mean_k, delta, x, U_old, p_old, r.U_X, r.best, dmax);
        fflush(stdout);
    }
    return 0;
}