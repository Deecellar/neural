const std = @import("std");

pub fn sigmoidDerivative(allocator: std.mem.Allocator, x: []f64) ![]f64 {
    var result = try allocator.alloc([]f64, x.len);
    for (x, 0..) |item, i| {
        // Exponential sigmoid derivative
        result[i] = 1 / (1 + @exp(-item));
        result[i] = result[i] * (1 - result[i]);
    }

    return result;
}

pub fn reluDerivative(allocator: std.mem.Allocator, x: []f64) ![]f64 {
    var result = try allocator.alloc(f64, x.len);
    for (x, 0..) |item, i| {
        if (item > 0) {
            result[i] = 0.01 * item;
        } else {
            result[i] = item;
        }
    }

    return result;
}

pub fn gaussianRadialBasisDerivative(allocator: std.mem.Allocator, x: []f64, centers: []f64, spread: []f64) ![]f64 {
    var result = try allocator.alloc(f64, x.len);
    for (x, 0..) |item, i| {
        result[i] = @exp(-((item - centers[i]) * (item - centers[i]) * spread[i % spread.len]));
    }
    return result;
}

// We create a center function that samples the input data and returns the center of the gaussian radial basis function
pub fn center(allocator: std.mem.Allocator, x: []f64) ![]f64 {
    var centers = try allocator.alloc(f64, x.len);
    var rand = std.rand.DefaultPrng.init(@bitCast(u64, std.time.milliTimestamp()));

    for (centers, 0..) |_, i| {
        centers[i] = x[rand.random().intRangeAtMostBiased(usize, 0, x.len - 1)];
    }
    return centers;
}

// K means clustering
pub fn kMeans(allocator: std.mem.Allocator, x: []f64, k: usize) ![]f64 {
    var centers = try center(allocator, x);

    var clusters = try allocator.alloc(usize, x.len);
    var newClusters = try allocator.alloc(usize, x.len);
    var clusterCounts = try allocator.alloc(usize, k);
    var clusterSums = try allocator.alloc(f64, k);
    defer {
        allocator.free(clusters);
        allocator.free(newClusters);
        allocator.free(clusterCounts);
        allocator.free(clusterSums);
    }
    while (true) {
        for (x, 0..) |item, i| {
            var minDistance: f64 = std.math.f64_max;
            var minIndex: usize = 0;
            for (centers, 0..) |c, j| {
                // Lets use euclidean distance
                var distance = @sqrt((item - c) * (item - c));
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = j;
                }
            }
            newClusters[i] = minIndex;
        }

        var changed = false;
        for (newClusters, 0..) |item, i| {
            if (item != clusters[i]) {
                changed = true;
                break;
            }
        }

        if (!changed) {
            break;
        }

        for (clusterCounts, 0..) |_, i| {
            clusterCounts[i] = 0;
        }

        for (clusterSums, 0..) |_, i| {
            clusterSums[i] = 0;
        }

        for (newClusters, 0..) |item, i| {
            clusterCounts[item] += 1;
            clusterSums[item] += x[i];
        }

        for (centers, 0..) |_, i| {
            centers[i] = clusterSums[i] / @intToFloat(f64, clusterCounts[i]);
        }

        for (clusters, 0..) |_, i| {
            clusters[i] = newClusters[i];
        }
    }
    // Replace any nan values with a random sample
    var rand = std.rand.DefaultPrng.init(@bitCast(u64, std.time.milliTimestamp()));
    for (centers, 0..) |item, i| {
        if (std.math.isNan(item) or std.math.isInf(item) or std.math.isSignalNan(item)) {
            centers[i] = x[rand.random().intRangeAtMostBiased(usize, 0, x.len - 1)];
        }
    }
    return centers;
}

pub fn gradientDescent(allocator: std.mem.Allocator, x: []f64, y: []f64, weights: []f64, bias: f64, learningRate: f64) ![]f64 {
    var result = try allocator.alloc(f64, weights.len);
    var biasResult: f64 = 0;
    defer {
        allocator.free(result);
    }
    for (x, 0..) |item, i| {
        var prediction = bias;
        for (weights) |w| {
            prediction += item * w;
        }
        var err = prediction - y[i];
        biasResult += err;
        for (weights, 0..) |_, j| {
            result[j] += err * item;
        }
    }
    biasResult /= @intToFloat(f64, x.len);
    for (result, 0..) |*item, i| {
        item.* /= @intToFloat(f64, x.len);
        item.* *= learningRate;
        item.* += weights[i];
    }
    biasResult *= learningRate;
    biasResult += bias;
    return result;
}
