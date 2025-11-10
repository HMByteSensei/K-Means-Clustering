#include <vector>    // For std::vector
#include <cmath>     // For std::sqrt and std::pow
#include <limits>    // For std::numeric_limits
#include <random>    // For modern C++ random number generation
#include <algorithm> // For std::shuffle
#include <chrono>
#include <iostream>
#include <omp.h>

/**
 * @struct Point
 * @brief A simple structure to represent a 2D point with x and y coordinates.
 */
struct Point {
    double x = 0.0;
    double y = 0.0;
};

/**
 * @struct Cluster
 * @brief Represents a cluster in the K-Means algorithm.
 *
 * Each cluster has an ID, a centroid (which is the mean of all points in it),
 * and a list of points assigned to it.
 */
struct Cluster {
    int id;
    Point centroid;
    std::vector<Point> points;

    /**
     * @brief Constructs a new Cluster.
     * @param cluster_id The unique identifier for the cluster.
     * @param c The initial centroid for the cluster.
     */
    Cluster(int cluster_id, Point c) : id(cluster_id), centroid(c) {}

    /**
     * @brief Clears the list of points assigned to this cluster.
     */
    void clear_points() {
        points.clear();
    }

    /**
     * @brief Adds a point to this cluster.
     * @param p The point to add.
     */
    void add_point(const Point& p) {
        points.push_back(p);
    }
};

/**
 * @class KMeans
 * @brief Encapsulates the logic for the K-Means clustering algorithm.
 */
class KMeans {
public:
    /**
     * @brief Constructs the KMeans algorithm object.
     * @param k The number of clusters to form (the "K" in K-Means).
     * @param max_iterations The maximum number of iterations to run before stopping.
     */
    KMeans(int k, int max_iterations = 100) 
        : num_clusters(k), max_iterations(max_iterations) {}

    /**
     * @brief Runs the K-Means algorithm on a given dataset of 2D points.
     * @param points The dataset of points to cluster.
     * @return A vector of Cluster objects, each containing its final centroid and assigned points.
     */
    std::vector<Cluster> run(const std::vector<Point>& points) {
        // Cannot have more clusters than points.
        if (points.size() < num_clusters) {
            // In a real-world scenario, you might throw an exception.
            // For this example, we return an empty result.
            return {};
        }

        // 1. Initialization: Randomly select K points as initial centroids.
        initialize_centroids(points);

        // This vector will store the cluster ID for each point.
        std::vector<int> point_assignments(points.size(), -1);

        // Main K-Means loop
        for (int iter = 0; iter < max_iterations; ++iter) {
            bool assignments_changed = false;

            // 2. Assignment Step: Assign each point to the nearest centroid.
            for (size_t i = 0; i < points.size(); ++i) {
                int nearest_cluster_id = get_nearest_cluster_id(points[i]);
                
                // If the point's cluster assignment has changed, note it.
                if (point_assignments[i] != nearest_cluster_id) {
                    point_assignments[i] = nearest_cluster_id;
                    assignments_changed = true;
                }
            }

            // 3. Update Step: Recalculate centroids based on new assignments.
            update_centroids(points, point_assignments);

            // 4. Convergence Check: If no assignments changed, the algorithm has converged.
            if (!assignments_changed) {
                break;
            }
        }

        // Final population of cluster points for the return value
        populate_final_clusters(points, point_assignments);
        return clusters;
    }

    std::vector<Cluster> run_with_parallelization(const std::vector<Point>& points) {
        // Cannot have more clusters than points.
        if (points.size() < num_clusters) {
            // In a real-world scenario, you might throw an exception.
            // For this example, we return an empty result.
            return {};
        }

        // 1. Initialization: Randomly select K points as initial centroids.
        initialize_centroids(points);

        // This vector will store the cluster ID for each point.
        std::vector<int> point_assignments(points.size(), -1);

        // Main K-Means loop
        for (int iter = 0; iter < max_iterations; ++iter) {
            bool assignments_changed = false;

            // 2. Assignment Step: Assign each point to the nearest centroid.
            #pragma omp parallel for
            for (size_t i = 0; i < points.size(); ++i) {
                int nearest_cluster_id = get_nearest_cluster_id(points[i]);
                
                // If the point's cluster assignment has changed, note it.
                if (point_assignments[i] != nearest_cluster_id) {
                    point_assignments[i] = nearest_cluster_id;
                    #pragma omp atomic write
                    assignments_changed = true;
                }
            }

            // 3. Update Step: Recalculate centroids based on new assignments.
            update_centroids_with_parallelisation(points, point_assignments);

            // 4. Convergence Check: If no assignments changed, the algorithm has converged.
            if (!assignments_changed) {
                break;
            }
        }

        // Final population of cluster points for the return value
        populate_final_clusters(points, point_assignments);
        return clusters;
    }

private:
    int num_clusters; // K
    int max_iterations;
    std::vector<Cluster> clusters;

    /**
     * @brief Initializes centroids using the Forgy method (randomly choosing K points from the dataset).
     * @param points The dataset.
     */
    void initialize_centroids(const std::vector<Point>& points) {
        clusters.clear();
        std::vector<Point> shuffled_points = points;

        // Use a modern C++ random number generator for shuffling
        //std::random_device rd;
        //std::mt19937 g(rd());
         std::mt19937 g(1234);
        std::shuffle(shuffled_points.begin(), shuffled_points.end(), g);

        // Select the first K unique points as initial centroids
        for (int i = 0; i < num_clusters; ++i) {
            clusters.emplace_back(i, shuffled_points[i]);
        }
    }

    /**
     * @brief Calculates the squared Euclidean distance between two points.
     *        (Using squared distance is a common optimization as it avoids the
     *        costly sqrt operation and yields the same comparison results.)
     * @param p1 The first point.
     * @param p2 The second point.
     * @return The squared Euclidean distance.
     */
    double calculate_squared_distance(const Point& p1, const Point& p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return dx * dx + dy * dy;
    }

    /**
     * @brief Finds the ID of the cluster with the centroid closest to a given point.
     * @param point The point to find the nearest cluster for.
     * @return The ID of the nearest cluster.
     */
    int get_nearest_cluster_id(const Point& point) {
        double min_dist_sq = std::numeric_limits<double>::max();
        int nearest_cluster_id = -1;

        for (const auto& cluster : clusters) {
            double dist_sq = calculate_squared_distance(point, cluster.centroid);
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_cluster_id = cluster.id;
            }
        }
        return nearest_cluster_id;
    }

    /**
     * @brief Updates the centroid of each cluster by calculating the mean of all points assigned to it.
     * @param points The full dataset of points.
     * @param assignments A vector mapping each point index to its assigned cluster ID.
     */
    void update_centroids(const std::vector<Point>& points, const std::vector<int>& assignments) {
        // Create vectors to accumulate sums and count points for each cluster
        std::vector<Point> new_centroids(num_clusters, {0.0, 0.0});
        std::vector<int> points_in_cluster(num_clusters, 0);

        for (size_t i = 0; i < points.size(); ++i) {
            int cluster_id = assignments[i];
            new_centroids[cluster_id].x += points[i].x;
            new_centroids[cluster_id].y += points[i].y;
            points_in_cluster[cluster_id]++;
        }

        // Calculate the new mean (centroid) for each cluster
        for (int i = 0; i < num_clusters; ++i) {
            // Avoid division by zero for empty clusters
            if (points_in_cluster[i] > 0) {
                clusters[i].centroid.x = new_centroids[i].x / points_in_cluster[i];
                clusters[i].centroid.y = new_centroids[i].y / points_in_cluster[i];
            }
            // Note: Handling empty clusters is a design choice. A more advanced
            // implementation might re-initialize the centroid of an empty cluster.
            // Here, we simply let it remain in its last known position.
        }
    }

    void update_centroids_with_parallelisation(const std::vector<Point>& points, const std::vector<int>& assignments) {
        // Create vectors to accumulate sums and count points for each cluster
        std::vector<Point> new_centroids(num_clusters, {0.0, 0.0});
        std::vector<int> points_in_cluster(num_clusters, 0);

        #pragma omp parallel
        {
            std::vector<Point> new_centroids_local(num_clusters, {0.0, 0.0});
            std::vector<int> points_in_cluster_local(num_clusters, 0);

            #pragma omp for
            for (size_t i = 0; i < points.size(); ++i)
            {
                int cluster_id = assignments[i];
                new_centroids_local[cluster_id].x += points[i].x;
                new_centroids_local[cluster_id].y += points[i].y;
                points_in_cluster_local[cluster_id]++;
            }
            #pragma omp critical
            {
                for (int i = 0; i < num_clusters; i++)
                {
                    new_centroids[i].x += new_centroids_local[i].x;
                    new_centroids[i].y += new_centroids_local[i].y;
                    points_in_cluster[i] += points_in_cluster_local[i];
                }
            }
        }

        // Calculate the new mean (centroid) for each cluster
        for (int i = 0; i < num_clusters; ++i) {
            // Avoid division by zero for empty clusters
            if (points_in_cluster[i] > 0) {
                clusters[i].centroid.x = new_centroids[i].x / points_in_cluster[i];
                clusters[i].centroid.y = new_centroids[i].y / points_in_cluster[i];
            }
        }
    }

    /**
     * @brief Populates the `points` vector of each cluster based on the final assignments.
     * @param points The full dataset of points.
     * @param assignments The final vector mapping each point index to its assigned cluster ID.
     */
    void populate_final_clusters(const std::vector<Point>& points, const std::vector<int>& assignments) {
        for (auto& cluster : clusters) {
            cluster.clear_points();
        }
        for (size_t i = 0; i < points.size(); ++i) {
            clusters[assignments[i]].add_point(points[i]);
        }
    }
};


// Include your KMeans implementation here
// (You can just paste your full KMeans code above this main function)

int main() {
    // Parameters
    const int num_points = 1000000;   // number of data points
    const int num_clusters = 10;     // number of clusters (K)

    // Generate random 2D points
    std::vector<Point> points;
    points.reserve(num_points);

    std::mt19937 gen(1234); 
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (int i = 0; i < num_points; ++i) {
        points.push_back({dist(gen), dist(gen)});
    }

    // Create KMeans object
    KMeans kmeans(num_clusters);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    //std::vector<Cluster> clusters = kmeans.run(points);
    std::vector<Cluster> clusters = kmeans.run_with_parallelization(points);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;

    // Output results
    std::cout << "K-Means finished in " << duration.count() << " seconds.\n";
    std::cout << "Final cluster centroids:\n";

    for (const auto& cluster : clusters) {
        std::cout << "Cluster " << cluster.id << ": ("
                  << cluster.centroid.x << ", "
                  << cluster.centroid.y << ")"
                  << " -> " << cluster.points.size() << " points\n";
    }

    return 0;
}

