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
