#pragma GCC target("avx")
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <immintrin.h> // OBAVEZNO: Biblioteka za AVX intrinzične funkcije


struct Point
{
    double x = 0.0;
    double y = 0.0;
};

struct Cluster
{
    int id;
    Point centroid;
    std::vector<Point> points;

    Cluster(int cluster_id, Point c) : id(cluster_id), centroid(c) {}

    void clear_points()
    {
        points.clear();
    }

    void add_point(const Point &p)
    {
        points.push_back(p);
    }
};

class KMeans
{
public:
    KMeans(int k, int max_iterations = 100)
        : num_clusters(k), max_iterations(max_iterations) {}

    std::vector<Cluster> run_with_parallelization(const std::vector<Point> &points)
    {
        if (points.size() < num_clusters) return {};

        initialize_centroids(points);
        std::vector<int> point_assignments(points.size(), -1);

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            bool assignments_changed = false;

            // 1. Assignment Step (Parallel)
            #pragma omp parallel for reduction(|:assignments_changed)
            for (size_t i = 0; i < points.size(); ++i)
            {
                int nearest_cluster_id = get_nearest_cluster_id(points[i]);
                if (point_assignments[i] != nearest_cluster_id)
                {
                    point_assignments[i] = nearest_cluster_id;
                    assignments_changed = true;
                }
            }

            // 2. Update Step (AVX OPTIMIZOVANO)
            // Ovo je dio koji je asistent tražio (linija ~280 u originalu)
            update_centroids_avx(points, point_assignments);

            if (!assignments_changed) break;
        }

        populate_final_clusters(points, point_assignments);
        return clusters;
    }

    // Originalni run metod (zadržan radi kompatibilnosti)
    std::vector<Cluster> run(const std::vector<Point> &points)
    {
        if (points.size() < num_clusters) return {};
        initialize_centroids(points);
        std::vector<int> point_assignments(points.size(), -1);

        for (int iter = 0; iter < max_iterations; ++iter)
        {
            bool assignments_changed = false;
            for (size_t i = 0; i < points.size(); ++i)
            {
                int nearest_cluster_id = get_nearest_cluster_id(points[i]);
                if (point_assignments[i] != nearest_cluster_id)
                {
                    point_assignments[i] = nearest_cluster_id;
                    assignments_changed = true;
                }
            }
            update_centroids(points, point_assignments);
            if (!assignments_changed) break;
        }
        populate_final_clusters(points, point_assignments);
        return clusters;
    }

private:
    int num_clusters;
    int max_iterations;
    std::vector<Cluster> clusters;

    // --- OVO JE KLJUČNA FUNKCIJA KOJU JE ASISTENT TRAŽIO ---
    void update_centroids_avx(const std::vector<Point> &points, const std::vector<int> &assignments)
    {
        int num_avx_clusters = (num_clusters + 1) / 2;

        std::vector<Point> global_sums(num_clusters, {0.0, 0.0});
        std::vector<int> global_counts(num_clusters, 0);

        #pragma omp parallel
        {
            // 1. ALOKACIJA: Koristimo _mm_malloc umjesto std::vector da osiguramo 32-byte alignment
            __m256d* local_sums_avx = (__m256d*)_mm_malloc(num_avx_clusters * sizeof(__m256d), 32);
            
            // Inicijalizacija na nulu (obavezno jer malloc ne briše smeće)
            for(int k = 0; k < num_avx_clusters; k++) {
                local_sums_avx[k] = _mm256_setzero_pd();
            }

            std::vector<int> local_counts(num_clusters, 0);
            const double* raw_points = reinterpret_cast<const double*>(points.data());
            size_t n = points.size();

            #pragma omp for
            for (size_t i = 0; i < n / 2 * 2; i += 2)
            {
                __m256d p_vec = _mm256_loadu_pd(&raw_points[i * 2]);
                __m128d p1_128 = _mm256_extractf128_pd(p_vec, 0);
                __m128d p2_128 = _mm256_extractf128_pd(p_vec, 1);

                int id1 = assignments[i];
                int id2 = assignments[i+1];

                local_counts[id1]++;
                local_counts[id2]++;

                // Tačka 1
                int vec_idx1 = id1 / 2;
                bool is_odd1 = id1 % 2;
                __m256d val1;
                if (is_odd1) {
                     val1 = _mm256_set_m128d(p1_128, _mm_setzero_pd());
                } else {
                     val1 = _mm256_set_m128d(_mm_setzero_pd(), p1_128);
                }
                local_sums_avx[vec_idx1] = _mm256_add_pd(local_sums_avx[vec_idx1], val1);

                // Tačka 2
                int vec_idx2 = id2 / 2;
                bool is_odd2 = id2 % 2;
                __m256d val2;
                if (is_odd2) {
                     val2 = _mm256_set_m128d(p2_128, _mm_setzero_pd());
                } else {
                     val2 = _mm256_set_m128d(_mm_setzero_pd(), p2_128);
                }
                local_sums_avx[vec_idx2] = _mm256_add_pd(local_sums_avx[vec_idx2], val2);
            }

            // Ostatak (neparan broj tačaka)
            if (n % 2 != 0) {
                #pragma omp single 
                {
                   // Ovaj dio mora biti single ili handled carefully u paralelizaciji ako loop ne pokriva sve
                   // Ali pošto je 'omp for' gore, onaj thread koji dobije zadnji chunk će ovo odraditi
                   // Mada, najsigurnije je ostatak obraditi van omp for-a, ali unutar parallel regiona samo ako je thread zadužen za taj index.
                   // Zbog jednostavnosti HPC zadatka, pretpostavimo da 'omp for' hendla iteracije, 
                   // ali moramo ručno hendlati zadnji element ako ga loop preskoči.
                   // NAJSIGURNIJE: Obraditi zadnju tačku sekvencijalno VAN parallel regiona ili unutar 'single'.
                   // Ovdje ćemo samo preskočiti za demo da ne komplikujemo logiku niti,
                   // jer AVX obično traži padding podataka.
                }
                // Za potrebe zadatka paralelizacije: ignorisat ćemo 1 tačku ako je neparan broj
                // ili se to rješava paddingom niza points na paran broj prije poziva funkcije.
            }

            #pragma omp critical
            {
                for (int j = 0; j < num_avx_clusters; ++j) {
                    double temp[4];
                    _mm256_storeu_pd(temp, local_sums_avx[j]);

                    int c_id_A = j * 2;
                    if (c_id_A < num_clusters) {
                        global_sums[c_id_A].x += temp[0];
                        global_sums[c_id_A].y += temp[1];
                        global_counts[c_id_A] += local_counts[c_id_A];
                    }

                    int c_id_B = j * 2 + 1;
                    if (c_id_B < num_clusters) {
                        global_sums[c_id_B].x += temp[2];
                        global_sums[c_id_B].y += temp[3];
                        global_counts[c_id_B] += local_counts[c_id_B];
                    }
                }
            }

            // 2. ČIŠĆENJE: Obavezno osloboditi memoriju!
            _mm_free(local_sums_avx);
        }

        // Ako je ostala 1 tačka (neparan broj), dodaj je ručno u globalne sume
        if (points.size() % 2 != 0) {
             size_t last_idx = points.size() - 1;
             int cid = assignments[last_idx];
             global_sums[cid].x += points[last_idx].x;
             global_sums[cid].y += points[last_idx].y;
             global_counts[cid]++;
        }

        calculate_new_mean(global_sums, global_counts);
    }
    
    void initialize_centroids(const std::vector<Point> &points)
    {
        clusters.clear();
        std::vector<Point> shuffled_points = points;
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(shuffled_points.begin(), shuffled_points.end(), g);
        for (int i = 0; i < num_clusters; ++i)
            clusters.emplace_back(i, shuffled_points[i]);
    }

    double calculate_squared_distance(const Point &p1, const Point &p2)
    {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return dx * dx + dy * dy;
    }

    int get_nearest_cluster_id(const Point &point)
    {
        double min_dist_sq = std::numeric_limits<double>::max();
        int nearest_cluster_id = -1;
        for (const auto &cluster : clusters)
        {
            double dist_sq = calculate_squared_distance(point, cluster.centroid);
            if (dist_sq < min_dist_sq)
            {
                min_dist_sq = dist_sq;
                nearest_cluster_id = cluster.id;
            }
        }
        return nearest_cluster_id;
    }

    // Stari update metod (zadržan ako treba)
    void update_centroids(const std::vector<Point> &points, const std::vector<int> &assignments)
    {
        std::vector<Point> new_centroids(num_clusters, {0.0, 0.0});
        std::vector<int> points_in_cluster(num_clusters, 0);
        for (size_t i = 0; i < points.size(); ++i)
        {
            int cluster_id = assignments[i];
            new_centroids[cluster_id].x += points[i].x;
            new_centroids[cluster_id].y += points[i].y;
            points_in_cluster[cluster_id]++;
        }
        calculate_new_mean(new_centroids, points_in_cluster);
    }

    void calculate_new_mean(std::vector<Point> new_centroids, std::vector<int> points_in_cluster){
        for (int i = 0; i < num_clusters; ++i)
        {
            if (points_in_cluster[i] > 0)
            {
                clusters[i].centroid.x = new_centroids[i].x / points_in_cluster[i];
                clusters[i].centroid.y = new_centroids[i].y / points_in_cluster[i];
            }
        }
    }

    void populate_final_clusters(const std::vector<Point> &points, const std::vector<int> &assignments)
    {
        for (auto &cluster : clusters) cluster.clear_points();
        for (size_t i = 0; i < points.size(); ++i)
            clusters[assignments[i]].add_point(points[i]);
    }
};

void save_clusters_to_csv(const std::vector<Cluster> &clusters, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) return;
    file << "x,y,cluster,type\n";
    for (const auto &cluster : clusters)
    {
        for (const auto &p : cluster.points)
            file << p.x << "," << p.y << "," << cluster.id << ",0\n";
        file << cluster.centroid.x << "," << cluster.centroid.y << "," << cluster.id << ",1\n";
    }
    file.close();
}

int main()
{
    const int num_points = 100000; // Povećao sam malo broj tačaka da se vidi razlika
    const int num_clusters = 7;

    std::vector<Point> points;
    points.reserve(num_points);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (int i = 0; i < num_points; ++i)
        points.push_back({dist(gen), dist(gen)});

    KMeans kmeans(num_clusters);

    // Mjerenje standardne verzije
    auto start = std::chrono::high_resolution_clock::now();
    kmeans.run(points); // Pokreće standardnu verziju (bez AVX update-a, samo serijski ili stari parallel)
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Mjerenje AVX verzije
    auto start_par = std::chrono::high_resolution_clock::now();
    auto clusters = kmeans.run_with_parallelization(points); // Pokreće AVX verziju
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end_par - start_par;

    std::cout << "Standard K-Means finished in " << duration.count() << " seconds.\n";
    std::cout << "AVX Optimized K-Means finished in " << duration_par.count() << " seconds.\n";

    save_clusters_to_csv(clusters, "clusters.csv");
    return 0;
}
