#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Undirected graph structure storing both adjacency list and edge list
struct Graph {
    int N = 0;
    int E = 0;
    vector<vector<int>> adj;
    vector<pair<int, int>> edges;
};

// Pack an undirected vertex pair (u, v) into a 64-bit integer key for O(1) edge lookup
inline uint64_t edge_key(int u, int v) {
    if (u > v) swap(u, v);
    return (static_cast<uint64_t>(u) << 32) | static_cast<uint32_t>(v);
}

// Load network from data/<language>_dependency_network.txt
Graph load_network(const string& filename) {
    ifstream in(filename);
    if (!in.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    int N_hdr, E_hdr;
    in >> N_hdr >> E_hdr;

    Graph g;
    g.N = N_hdr;
    g.adj.resize(N_hdr);
    g.edges.reserve(E_hdr);

    unordered_map<string, int> word_to_id;
    word_to_id.reserve(N_hdr * 2);

    string u_str, v_str;
    while (in >> u_str >> v_str) {
        auto it_u = word_to_id.find(u_str);
        int u = (it_u == word_to_id.end()) ? (word_to_id[u_str] = word_to_id.size()) : it_u->second;
        auto it_v = word_to_id.find(v_str);
        int v = (it_v == word_to_id.end()) ? (word_to_id[v_str] = word_to_id.size()) : it_v->second;

        if (u != v) {
            g.adj[u].push_back(v);
            g.adj[v].push_back(u);
            g.edges.push_back({u, v});
        }
    }
    g.E = g.edges.size();
    return g;
}

// Compute local closeness C_i for a single vertex using BFS
double compute_single_closeness(const Graph& g, int src, vector<int>& dist, vector<int>& q) {
    int k_src = g.adj[src].size();
    if (k_src == 0) return 0.0;

    // Degree-1 shortcut: deduce distances from the unique neighbor's BFS
    int bfs_root = (k_src == 1) ? g.adj[src][0] : src;
    if (k_src == 1 && g.adj[bfs_root].size() == 1) {
        return 1.0 / (g.N - 1); // Isolated dyad
    }

    int head = 0, tail = 0;
    q[tail++] = bfs_root;
    dist[bfs_root] = 0;

    int max_d = 0;
    vector<int> dist_freq(64, 0);

    while (head < tail) {
        int u = q[head++];
        int d = dist[u];
        int next_d = d + 1;
        for (int v : g.adj[u]) {
            if (dist[v] == -1) {
                dist[v] = next_d;
                q[tail++] = v;
                if (next_d >= (int)dist_freq.size()) dist_freq.resize(next_d * 2, 0);
                dist_freq[next_d]++;
                if (next_d > max_d) max_d = next_d;
            }
        }
    }

    // Reset visited distances for next call
    for (int i = 0; i < tail; ++i) dist[q[i]] = -1;

    double sum_inv = 0.0;
    if (k_src == 1) {
        sum_inv = 0.5;
        for (int d = 1; d <= max_d; ++d) {
            if (dist_freq[d] > 0) sum_inv += (double)dist_freq[d] / (d + 1);
        }
    } else {
        for (int d = 1; d <= max_d; ++d) {
            if (dist_freq[d] > 0) sum_inv += (double)dist_freq[d] / d;
        }
    }
    return sum_inv / (g.N - 1);
}

// Compute exact or sampled closeness centrality of the real graph
// If sample_Order is passed, uses the exact same M vertices for fair variance reduction
double compute_closeness(const Graph& g, const vector<int>& eval_nodes) {
    int M = eval_nodes.size();
    double total_c = 0.0;

    #pragma omp parallel reduction(+:total_c)
    {
        vector<int> dist(g.N, -1);
        vector<int> q(g.N);
        #pragma omp for schedule(dynamic, 32)
        for (int idx = 0; idx < M; ++idx) {
            total_c += compute_single_closeness(g, eval_nodes[idx], dist, q);
        }
    }
    return total_c / M;
}

// Evaluate whether C_NH >= C_target using tight 2-hop upper bounds and early stopping
bool is_null_closeness_geq(const Graph& g, double c_target, const vector<int>& sample_nodes,
                           bool use_exact_all, int& nodes_evaluated_out) {
    int N = g.N;
    if (!use_exact_all) {
        // Sampled mode (Section 3.2) with Section 5 2-hop upper-bound early stopping
        int M = sample_nodes.size();
        vector<int> order = sample_nodes;
        sort(order.begin(), order.end(), [&](int a, int b) {
            return g.adj[a].size() > g.adj[b].size();
        });

        // Precompute tight 2-hop upper bound u_i for the M sampled nodes
        vector<double> u_bound(M, 0.0);
        double sum_upper_rem = 0.0;
        for (int idx = 0; idx < M; ++idx) {
            int u = order[idx];
            long long k1 = g.adj[u].size();
            if (k1 == 0) {
                u_bound[idx] = 0.0;
                continue;
            }
            long long k2_max = 0;
            for (int v : g.adj[u]) k2_max += (long long)g.adj[v].size() - 1;
            long long rem1 = (N - 1) - k1;
            long long k2 = min(rem1, k2_max);
            long long k3 = rem1 - k2;
            u_bound[idx] = (k1 + 0.5 * k2 + (1.0 / 3.0) * k3) / (N - 1);
            sum_upper_rem += u_bound[idx];
        }

        double sum_exact = 0.0;
        double target_sum = c_target * M;
        int evaluated = 0;
        bool decided = false;
        bool result = false;

        // Evaluate in parallel batches to check early stopping conditions
        int batch_size = 64;
        for (int start = 0; start < M && !decided; start += batch_size) {
            int end = min(M, start + batch_size);
            double batch_exact = 0.0;
            double batch_upper = 0.0;

            #pragma omp parallel reduction(+:batch_exact, batch_upper)
            {
                vector<int> dist(N, -1);
                vector<int> q(N);
                #pragma omp for schedule(static)
                for (int idx = start; idx < end; ++idx) {
                    batch_exact += compute_single_closeness(g, order[idx], dist, q);
                    batch_upper += u_bound[idx];
                }
            }
            sum_exact += batch_exact;
            sum_upper_rem -= batch_upper;
            evaluated = end;

            if (sum_exact >= target_sum - 1e-12) {
                result = true;
                decided = true;
            } else if (sum_exact + sum_upper_rem < target_sum - 1e-12) {
                result = false;
                decided = true;
            }
        }
        nodes_evaluated_out = evaluated;
        if (!decided) result = (sum_exact >= target_sum - 1e-12);
        return result;
    }

    // Full Exact Mode (M = N) with k=0, k=1 leaf deduction and 2-hop bounds
    vector<int> comp_id(N, -1), comp_size, q_bfs(N);
    for (int i = 0; i < N; ++i) {
        if (comp_id[i] == -1) {
            int cid = comp_size.size();
            int h = 0, t = 0;
            q_bfs[t++] = i;
            comp_id[i] = cid;
            while (h < t) {
                int u = q_bfs[h++];
                for (int v : g.adj[u]) {
                    if (comp_id[v] == -1) {
                        comp_id[v] = cid;
                        q_bfs[t++] = v;
                    }
                }
            }
            comp_size.push_back(t);
        }
    }

    vector<double> u_bound(N, 0.0);
    vector<char> resolved(N, 0);
    vector<vector<int>> leaves_of(N);
    double sum_exact = 0.0;
    double sum_upper_rem = 0.0;

    for (int i = 0; i < N; ++i) {
        long long k1 = g.adj[i].size();
        if (k1 == 0) {
            resolved[i] = 1;
            continue;
        }
        if (k1 == 1) {
            int p = g.adj[i][0];
            if (g.adj[p].size() == 1) {
                resolved[i] = 1;
                sum_exact += 1.0 / (N - 1);
                continue;
            }
            leaves_of[p].push_back(i);
        }
        long long S = comp_size[comp_id[i]];
        long long k2_max = 0;
        for (int v : g.adj[i]) k2_max += (long long)g.adj[v].size() - 1;
        long long rem1 = (S - 1) - k1;
        long long k2 = min(rem1, k2_max);
        long long k3 = rem1 - k2;
        u_bound[i] = (k1 + 0.5 * k2 + (1.0 / 3.0) * k3) / (N - 1);
        sum_upper_rem += u_bound[i];
    }

    vector<int> hubs;
    for (int i = 0; i < N; ++i) {
        if (!resolved[i] && g.adj[i].size() >= 2) hubs.push_back(i);
    }
    sort(hubs.begin(), hubs.end(), [&](int a, int b) {
        return g.adj[a].size() > g.adj[b].size();
    });

    double target_sum = c_target * N;
    int H = hubs.size();
    int evaluated = 0;
    bool decided = false, result = false;
    int batch_size = 64;

    for (int start = 0; start < H && !decided; start += batch_size) {
        int end = min(H, start + batch_size);
        double batch_exact = 0.0;
        double batch_upper = 0.0;

        #pragma omp parallel reduction(+:batch_exact, batch_upper)
        {
            vector<int> dist(N, -1), q(N);
            vector<int> dist_freq(64, 0);
            #pragma omp for schedule(dynamic, 4)
            for (int idx = start; idx < end; ++idx) {
                int root = hubs[idx];
                int h = 0, t = 0, max_d = 0;
                q[t++] = root;
                dist[root] = 0;
                while (h < t) {
                    int u = q[h++];
                    int nd = dist[u] + 1;
                    for (int v : g.adj[u]) {
                        if (dist[v] == -1) {
                            dist[v] = nd;
                            q[t++] = v;
                            if (nd >= (int)dist_freq.size()) dist_freq.resize(nd * 2, 0);
                            dist_freq[nd]++;
                            if (nd > max_d) max_d = nd;
                        }
                    }
                }
                double c_root = 0.0;
                for (int d = 1; d <= max_d; ++d) {
                    if (dist_freq[d] > 0) c_root += (double)dist_freq[d] / d;
                }
                c_root /= (N - 1);
                batch_exact += c_root;
                batch_upper += u_bound[root];

                if (!leaves_of[root].empty()) {
                    double c_leaf = 0.5;
                    for (int d = 1; d <= max_d; ++d) {
                        if (dist_freq[d] > 0) c_leaf += (double)dist_freq[d] / (d + 1);
                    }
                    c_leaf /= (N - 1);
                    batch_exact += c_leaf * leaves_of[root].size();
                    for (int lf : leaves_of[root]) batch_upper += u_bound[lf];
                }
                for (int d = 1; d <= max_d; ++d) dist_freq[d] = 0;
                for (int i = 0; i < t; ++i) dist[q[i]] = -1;
            }
        }
        sum_exact += batch_exact;
        sum_upper_rem -= batch_upper;
        evaluated = end;

        if (sum_exact >= target_sum - 1e-12) {
            result = true;
            decided = true;
        } else if (sum_exact + sum_upper_rem < target_sum - 1e-12) {
            result = false;
            decided = true;
        }
    }
    nodes_evaluated_out = evaluated;
    if (!decided) result = (sum_exact >= target_sum - 1e-12);
    return result;
}

// Generate an Erdos-Renyi random graph G(N, E) without loops or multiedges
Graph generate_erdos_renyi(int N, int E, mt19937_64& rng) {
    Graph g;
    g.N = N;
    g.E = E;
    g.adj.resize(N);
    g.edges.reserve(E);

    unordered_set<uint64_t> existing;
    existing.reserve(E * 2);
    uniform_int_distribution<int> dist(0, N - 1);

    while ((int)g.edges.size() < E) {
        int u = dist(rng);
        int v = dist(rng);
        if (u == v) continue;
        uint64_t key = edge_key(u, v);
        if (existing.insert(key).second) {
            g.adj[u].push_back(v);
            g.adj[v].push_back(u);
            g.edges.push_back({u, v});
        }
    }
    return g;
}

// Generate a degree-preserving randomized graph via the Switching Model (Q * E trials)
Graph generate_switching_graph(const Graph& orig, int Q, mt19937_64& rng) {
    Graph g;
    g.N = orig.N;
    g.E = orig.E;
    g.edges = orig.edges;

    int E = g.E;
    unordered_set<uint64_t> existing;
    existing.reserve(E * 2);
    for (const auto& e : g.edges) {
        existing.insert(edge_key(e.first, e.second));
    }

    long long total_trials = (long long)Q * E;
    uniform_int_distribution<int> edge_dist(0, E - 1);

    for (long long t = 0; t < total_trials; ++t) {
        int idx1 = edge_dist(rng);
        int idx2 = edge_dist(rng);
        if (idx1 == idx2) continue; // Coincident edges (Case 1)

        int u = g.edges[idx1].first;
        int v = g.edges[idx1].second;
        int s = g.edges[idx2].first;
        int w = g.edges[idx2].second;

        // Randomly orient the second edge to allow both degree-preserving swaps equally
        if (t & 1) swap(s, w);

        // Check for coincident vertices (Case 2: prevents self-loops and trivial swaps)
        if (u == s || u == w || v == s || v == w) continue;

        // Check for multiedges (Case 3)
        uint64_t new_k1 = edge_key(u, s);
        uint64_t new_k2 = edge_key(v, w);
        if (existing.count(new_k1) || existing.count(new_k2)) continue;

        // Perform valid switch
        existing.erase(edge_key(u, v));
        existing.erase(edge_key(s, w));
        existing.insert(new_k1);
        existing.insert(new_k2);

        g.edges[idx1] = {u, s};
        g.edges[idx2] = {v, w};
    }

    // Rebuild adjacency list once after all Q*E switching trials finish
    g.adj.assign(g.N, vector<int>());
    for (int i = 0; i < g.N; ++i) g.adj[i].reserve(orig.adj[i].size());
    for (const auto& e : g.edges) {
        g.adj[e.first].push_back(e.second);
        g.adj[e.second].push_back(e.first);
    }
    return g;
}

int main(int argc, char* argv[]) {
    int T = (argc > 1) ? stoi(argv[1]) : 25;
    int Q = (argc > 2) ? stoi(argv[2]) : 10;
    double sample_ratio = (argc > 3) ? stod(argv[3]) : 0.05; // Default 5% sampling (or 1.0 for exact)
    string single_lang = (argc > 4) ? argv[4] : "ALL";

    vector<string> languages = {
        "Hungarian", "Greek", "Basque", "Arabic", "Catalan", "Chinese",
        "English", "Italian", "Spanish", "Polish", "Turkish", "Czech"
    };
    if (single_lang != "ALL") {
        languages = {single_lang};
    }

    ofstream out_csv("results/table_2_significance.csv");
    out_csv << "language,N,E,M_sampled,sample_ratio,closeness,p_binomial,p_switching,time_sec\n";

    cout << left << setw(12) << "Language"
         << right << setw(8) << "N"
         << setw(8) << "M"
         << setw(12) << "Closeness"
         << setw(12) << "p(Binom)"
         << setw(12) << "p(Switch)"
         << setw(10) << "Time(s)" << "\n";
    cout << string(74, '-') << "\n";

    mt19937_64 rng(42);

    for (const string& lang : languages) {
        auto t0 = chrono::high_resolution_clock::now();
        string path = "data/" + lang + "_dependency_network.txt";
        Graph g = load_network(path);

        bool use_exact = (sample_ratio >= 0.999);
        int M = use_exact ? g.N : max(500, min(g.N, (int)round(g.N * sample_ratio)));
        if (M == g.N) use_exact = true;

        // Uniform random permutation of vertices for Section 3.2 sampling
        vector<int> perm(g.N);
        for (int i = 0; i < g.N; ++i) perm[i] = i;
        if (!use_exact) {
            shuffle(perm.begin(), perm.end(), rng);
            perm.resize(M);
        }

        double c_real = compute_closeness(g, perm);

        int count_binom_geq = 0;
        int count_switch_geq = 0;

        for (int t = 0; t < T; ++t) {
            Graph g_er = generate_erdos_renyi(g.N, g.E, rng);
            int eval_er = 0;
            if (is_null_closeness_geq(g_er, c_real, perm, use_exact, eval_er)) {
                count_binom_geq++;
            }

            Graph g_sw = generate_switching_graph(g, Q, rng);
            int eval_sw = 0;
            if (is_null_closeness_geq(g_sw, c_real, perm, use_exact, eval_sw)) {
                count_switch_geq++;
            }
        }

        double p_binom = (double)count_binom_geq / T;
        double p_switch = (double)count_switch_geq / T;

        auto t1 = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();

        cout << left << setw(12) << lang
             << right << setw(8) << g.N
             << setw(8) << M
             << setw(12) << fixed << setprecision(6) << c_real
             << setw(12) << setprecision(4) << p_binom
             << setw(12) << setprecision(4) << p_switch
             << setw(10) << setprecision(2) << elapsed << "\n";

        out_csv << lang << "," << g.N << "," << g.E << "," << M << ","
                << (double)M / g.N << "," << setprecision(8) << c_real << ","
                << setprecision(4) << p_binom << "," << p_switch << ","
                << setprecision(2) << elapsed << "\n";
        out_csv.flush();
    }

    return 0;
}