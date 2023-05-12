//! A neural network layer

const std = @import("std");

pub const Layer = struct {
    weights: []f64,
    input_gradients: []f64 = undefined,
    input: usize,
    output: usize,
    last_input: []f64 = undefined,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !Layer {
        const size = input_size * output_size;
        const weights = try allocator.alloc(f64, size);
        return Layer{
            .weights = weights,
            .input = input_size,
            .output = output_size,
        };
    }

    pub fn forward(self: *Layer, allocator: std.mem.Allocator, input: []f64) ![]f64 {
        var outputs = try allocator.alloc(f64, self.output);
        for (0..self.output) |i| {
            var sum: f64 = 0.0;
            for (0..self.input) |j| {
                sum += input[j] * self.weights[i * self.input + j];
            }
            outputs[i] = sum;
        }
        self.last_input = input;
        return input;
    }

    pub fn backward(self: *Layer, allocator: std.mem.Allocator, input: []f64) ![]f64 {
        var weight = try allocator.alloc(f64, self.input);
        const batch = input.len / self.input;
        var input_gradients = try allocator.alloc(f64, batch * self.input);
        for (0..batch) |b| {
            for (0..self.input) |i| {
                for (0..self.output) |o| {
                    input_gradients[b * self.input + i] += self.weights[o * self.input + i] * input[b * self.output + o];
                    weight[i * self.output + o] += self.weights[o * self.input + i] * input[b * self.output + o];
                }
            }
        }
        self.input_gradients = input_gradients;
        self.weights = weight;
        return input_gradients;
    }

    pub fn update(self: *Layer, input: []f64) void {
        for (self.weights, 0..) |*w, i| {
            w.* -= 0.01 * input[i];
        }
    }
};
