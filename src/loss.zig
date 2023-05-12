//! Loss function for the neural network
//!
const std = @import("std");
pub fn mse(
    allocator: std.mem.Allocator,
    inputs: []f64,
    target: []f64,
) ![]f64 {
    var loss = try allocator.alloc(f64, target.len);
    for (loss, 0..) |*l, i| {
        l.* = (inputs[i] - target[i]) * (inputs[i] - target[i]);
        l.* /= @intToFloat(f64, target.len);
    }
    return loss;
}

pub fn mae(
    allocator: std.mem.Allocator,
    inputs: []f64,
    target: []f64,
) ![]f64 {
    var loss = try allocator.alloc(f64, target.len);
    for (loss, 0..) |*l, i| {
        l.* = @fabs(inputs[i] - target[i]);
        l.* /= @intToFloat(f64, target.len);
    }
    return loss;
}

pub fn huberLoss(
    allocator: std.mem.Allocator,
    inputs: []f64,
    target: []f64,
    delta: f64,
) ![]f64 {
    var loss = try allocator.alloc(f64, target.len);
    for (loss, 0..) |*l, i| {
        var diff = inputs[i] - target[i];
        if (diff > delta) {
            l.* = delta * (diff - delta / 2.0);
        } else if (diff < -delta) {
            l.* = -delta * (diff + delta / 2.0);
        } else {
            l.* = diff * diff / 2.0;
        }
        l.* /= @intToFloat(f64, target.len);
    }
    return loss;
}
