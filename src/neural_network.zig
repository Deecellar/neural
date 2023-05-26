const std = @import("std");
const layer = @import("layer.zig");
const loss = @import("loss.zig");
const activation_algorithmn = @import("algorithmn.zig");
pub const NeuralNetwork = struct {
    layer: std.ArrayListUnmanaged(layer.Layer),
    outputs: []f64,
    previousInputs: []f64,
    loss: std.ArrayListUnmanaged(f64),
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    pub fn init(allocator: std.mem.Allocator) !NeuralNetwork {
        var self: NeuralNetwork = undefined;
        self.allocator = allocator;
        self.layer = .{};
        self.loss = .{};
        self.arena = std.heap.ArenaAllocator.init(self.allocator);
        return self;
    }
    pub fn addLayer(self: *NeuralNetwork, inputs: usize, outputs: usize) !void {
        var lay = try layer.Layer.init(self.arena.allocator(), inputs, outputs);
        try self.layer.append(self.allocator, lay);
    }
    pub fn forward(self: *NeuralNetwork, inputs: []f64) ![]f64 {
        self.previousInputs = inputs;
        var outputs: []f64 = inputs;
        var local_arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
        defer local_arena.deinit();
        // We activate our radial neural network
        for (self.layer.items) |*lay| {
            outputs = try lay.forward(local_arena.allocator(), outputs);
            var centers = try activation_algorithmn.center(local_arena.allocator(), outputs);
            outputs = try activation_algorithmn.gaussianRadialBasisDerivative(local_arena.allocator(), outputs, centers, lay.weights);
        }
        self.outputs = try self.arena.allocator().dupe(f64, outputs);
        return outputs;
    }
    pub fn backward(self: *NeuralNetwork, targets: []f64) !void {
        var current_loss = targets;
        var index: usize = 0;
        var local_arena = std.heap.ArenaAllocator.init(self.allocator);
        defer local_arena.deinit();
        while (index < self.layer.items.len) {
            var lay = self.layer.items[self.layer.items.len - index - 1];
            current_loss = try lay.backward(local_arena.allocator(), current_loss);
            var centers = try activation_algorithmn.center(local_arena.allocator(), self.previousInputs);
            current_loss = try activation_algorithmn.gaussianRadialBasisDerivative(local_arena.allocator(), current_loss, centers, lay.weights);
            index += 1;
        }
        self.loss.clearRetainingCapacity();
        try self.loss.appendSlice(self.arena.allocator(), current_loss);
    }
    pub fn update(self: *NeuralNetwork) void {
        for (self.layer.items) |*lay| {
            lay.update(lay.weights);
        }
    }

    pub fn train(self: *NeuralNetwork, inputs: [][]f64, targets: [][]f64, epochs: usize, delta: f64) !void {
        for (0..epochs) |_| {
            for (inputs, 0..) |input, target| {
                var outputs = try self.forward(input);
                var lossing = try loss.huberLoss(self.arena.allocator(), outputs, targets[target], delta);
                try self.backward(lossing);
                self.update();
            }
        }
    }

    pub fn predict(self: *NeuralNetwork, inputs: [][]f64) ![][]f64 {
        var outputs: std.ArrayListUnmanaged([]f64) = .{};

        for (inputs) |input| {
            var output = try self.forward(input);
            try outputs.append(self.allocator, output);
        }
        return outputs.toOwnedSlice(self.allocator);
    }

    pub fn save(self: *NeuralNetwork, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path);
        var writer = std.io.BufferedWriter.init(file.writer());
        try writer.print("NeuralNetwork\n");
        try writer.print("Layers: {}\n", self.layer.items.len);
        for (self.layer.items) |lay| {
            try lay.save(writer);
        }
        try writer.flush();
    }

    pub fn load(self: *NeuralNetwork, path: []const u8, allocator: std.mem.Allocator) !void {
        var file = try std.fs.cwd().openFile(path, .{});
        self.allocator = allocator;
        self.layer = .{};
        self.loss = .{};
        var reader = std.io.BufferedReader.init(file.reader());
        var line = try reader.readLineAlloc(allocator);
        if (std.mem.eql(u8, line, "NeuralNetwork")) {
            if (@import("builtin").mode == .Debug) {
                std.log.err("Invalid file format");
            }
            return error.InvalidFileFormat;
        }
        line = try reader.readLineAlloc(allocator);
        var layers = try line.split(":", .{});
        if (layers.len != 2) {
            if (@import("builtin").mode == .Debug) {
                std.log.err("Invalid file format");
            }
            return error.InvalidFileFormat;
        }
        var layers_count = try std.fmt.parseInt(usize, line, 10);
        for (0..layers_count) |_| {
            line = try reader.readLineAlloc(allocator);
            var lay = try layer.Layer.load(line);
            try self.addLayer(lay);
        }
    }

    pub fn validate(self: *NeuralNetwork) bool {
        // The next layer should have the same number of inputs as the previous layer has outputs
        var previous_outputs = self.layer.items[0].output;
        for (self.layer.items, 1..) |lay, i| {
            if (lay.input != previous_outputs) {
                if (@import("builtin").mode == .Debug) {
                    std.log.err("Layer {} has {} inputs, but the previous layer has {} outputs", .{ i, lay.input, previous_outputs });
                }
                return false;
            }
            previous_outputs = lay.output;
        }
        return true;
    }
};
