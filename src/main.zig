const std = @import("std");
const csv = @import("csv_parser.zig");
const neural_network = @import("neural_network.zig");
pub fn main() !void {
    var process_dir = dir: {
        var process = try std.process.argsWithAllocator(std.heap.page_allocator);
        defer process.deinit();
        var exe_path = process.next() orelse return error.FileNotFound;
        var exe_dir = std.fs.path.dirname(exe_path) orelse return error.FileNotFound;
        break :dir try std.mem.concat(std.heap.page_allocator, u8, &.{ exe_dir, "/data" });
    };
    var files = try std.fs.openDirAbsolute(process_dir, .{});
    std.heap.page_allocator.free(process_dir);
    defer files.close();
    var stdout_file = std.io.getStdOut();
    defer stdout_file.close();
    var stdout = stdout_file.writer();
    var stdout_buffered_file = std.io.bufferedWriter(stdout);
    var stdout_buffered = stdout_buffered_file.writer();
    var file_buffer: [4 * 1024 * 1024]u8 = undefined;
    var experiment: [6]Experiment = undefined;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};

    defer if (gpa.deinit() == .leak) std.log.warn("Memory leak detected\n", .{});
    defer for (experiment) |exp| {
        gpa.allocator().free(exp.inputs);
        gpa.allocator().free(exp.outputs);
    };
    {
        var arena = std.heap.ArenaAllocator.init(gpa.allocator());
        var ally = arena.allocator();
        defer arena.deinit();
        var data_list = try std.ArrayList(Data).initCapacity(ally, 10000);
        defer data_list.deinit();
        inline for (.{ 100, 500, 1000, 2000, 5000, 10000 }, 0..) |v, i| {
            {
                var file_name = std.fmt.comptimePrint("data_r{d}.csv", .{v});
                var file = try files.openFile(file_name, .{});
                defer file.close();
                var data_read = try file.readAll(&file_buffer);
                var data = file_buffer[0..data_read];
                var parser = try csv.parseCsv(Data, data, null, true);
                // We print the file name in a pretty way with === on both sides
                while (parser.next()) |row| {
                    try data_list.append(row);
                }
            }
            // We separate input and output
            var input = try std.ArrayList(Input).initCapacity(gpa.allocator(), data_list.items.len);
            var output = try std.ArrayList(Output).initCapacity(gpa.allocator(), data_list.items.len);
            for (data_list.items) |data| {
                try input.append(Input{
                    .d1 = data.d1,
                    .d2 = data.d2,
                    .d3 = data.d3,
                    .d4 = data.d4,
                    .d5 = data.d5,
                });
                try output.append(Output{
                    .result = data.result,
                });
            }
            experiment[i] = Experiment.init(try input.toOwnedSlice(), try output.toOwnedSlice(), 0.8);
        }
    }
    var nn = try neural_network.NeuralNetwork.init(gpa.allocator());
    try nn.addLayer(5, 5);
    try nn.addLayer(5, 5);
    try nn.addLayer(5, 2);
    try nn.addLayer(2, 1);
    if (nn.validate()) {
        for (experiment) |e| {
            var ff: std.ArrayList([]f64) = std.ArrayList([]f64).init(gpa.allocator());
            var fl: std.ArrayList([]f64) = std.ArrayList([]f64).init(gpa.allocator());
            defer ff.deinit();
            defer fl.deinit();
            for (e.training.training_data) |d| {
                try ff.append(&[_]f64{ d.d1, d.d2, d.d3, d.d4, d.d5 });
            }
            for (e.training.training_labels) |d| {
                try fl.append(&[_]f64{d.result});
            }
            try nn.train(try ff.toOwnedSlice(), try fl.toOwnedSlice(), 1000, 0.1);
            ff.clearRetainingCapacity();
            fl.clearRetainingCapacity();
            for (e.testing.testing_data) |d| {
                try ff.append(&[_]f64{ d.d1, d.d2, d.d3, d.d4, d.d5 });
            }
            for (e.testing.testing_labels) |d| {
                try fl.append(&[_]f64{d.result});
            }
            var result = try nn.predict(try ff.toOwnedSlice());
            // We calculate the error
            var err: f64 = 0.0;
            for (result, 0..) |r, i| {
                err += (r[0] - fl.items[i][0]) * (r[0] - fl.items[i][0]);
            }
            err /= @intToFloat(f64, result.len);
            err = @sqrt(err);
            try stdout_buffered.print("Error for {d} data points: {any}\n", .{ e.inputs.len, err });
        }
    }
    try stdout_buffered_file.flush();
}

pub const Data = struct {
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
    result: f64,
};

pub const Input = struct {
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64,
    d5: f64,
};

pub const Output = struct {
    result: f64,
};

pub const Training = struct {
    training_data: []Input,
    training_labels: []Output,
};

pub const Testing = struct {
    testing_data: []Input,
    testing_labels: []Output,
};

pub const Experiment = struct {
    inputs: []Input,
    outputs: []Output,
    division: f64,
    training: Training,
    testing: Testing,
    pub fn init(input: []Input, output: []Output, division: f64) Experiment {
        var self: Experiment = undefined;
        self.inputs = input;
        self.outputs = output;
        self.division = division;
        var training_size = @floor(@intToFloat(f64, input.len) * division);
        self.training = Training{
            .training_data = input[0..@floatToInt(usize, training_size)],
            .training_labels = output[0..@floatToInt(usize, training_size)],
        };
        self.testing = Testing{
            .testing_data = input[@floatToInt(usize, training_size)..],
            .testing_labels = output[@floatToInt(usize, training_size)..],
        };
        return self;
    }
};
