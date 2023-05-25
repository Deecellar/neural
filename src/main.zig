const std = @import("std");
const csv = @import("csv_parser.zig");
const neural_network = @import("neural_network.zig");
const build_options = @import("build_options");
pub const learning_rate = 0.1;
pub fn main() !void {
    {
        // We get the current executable path via cwd and argv[0]
        var exe_cwd = std.fs.cwd();
        var args = try std.process.argsWithAllocator(std.heap.page_allocator);
        defer args.deinit();
        var exe_arg = args.next() orelse return error.FileNotFound;
        var exe_dir = try exe_cwd.openDir(std.fs.path.dirname(exe_arg) orelse ".", .{});
        defer exe_dir.close();
        try exe_dir.setAsCwd();
    }
    var files = std.fs.cwd();
    defer files.close();
    var stdout_file = std.io.getStdOut();
    defer stdout_file.close();
    var stdout = stdout_file.writer();
    var stdout_buffered_file = std.io.bufferedWriter(stdout);
    var stdout_buffered = stdout_buffered_file.writer();
    var file_buffer: [4 * 1024 * 1024]u8 = undefined;
    var experiment: [build_options.data.len]Experiment = undefined;
    // We create a file where we save the results
    var result_file = try files.createFile("result.csv", .{});
    defer result_file.close();
    var result_writer = result_file.writer();
    try result_writer.writeAll("neural_network_type, data_set_size, training_split,training_time(ns),error, mem_usage, learning_rate\n");
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
        inline for (build_options.data, 0..) |v, i| {
            {
                var file_name = std.fmt.comptimePrint("data/data_r{d}.csv", .{v});
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
            experiment[i] = Experiment.init(try input.toOwnedSlice(), try output.toOwnedSlice(), 0.7);
        }
    }
    var nn = try neural_network.NeuralNetwork.init(gpa.allocator());
    try nn.addLayer(5, 5);
    try nn.addLayer(5, 2);
    try nn.addLayer(2, 1);
    var last_mem: f64 = 0;
    if (nn.validate()) {
        for (experiment) |e| {
            var ff: std.ArrayList([]f64) = std.ArrayList([]f64).init(gpa.allocator());
            var fl: std.ArrayList([]f64) = std.ArrayList([]f64).init(gpa.allocator());
            defer ff.deinit();
            defer fl.deinit();
            for (e.training.training_data) |d| {
                var data = try gpa.allocator().alloc(f64, 5);
                data[0] = d.d1;
                data[1] = d.d2;
                data[2] = d.d3;
                data[3] = d.d4;
                data[4] = d.d5;
                try ff.append(data);
            }
            for (e.training.training_labels) |d| {
                var data = try gpa.allocator().alloc(f64, 1);
                data[0] = d.result;
                try fl.append(data);
            }
            var timer = try std.time.Timer.start();
            try nn.train(try ff.toOwnedSlice(), try fl.toOwnedSlice(), 1000, 0.1);
            var elapsed = timer.lap();
            for (e.testing.testing_data) |d| {
                var data = try gpa.allocator().alloc(f64, 5);
                data[0] = d.d1;
                data[1] = d.d2;
                data[2] = d.d3;
                data[3] = d.d4;
                data[4] = d.d5;
                try ff.append(data);
            }
            for (e.testing.testing_labels) |d| {
                var data = try gpa.allocator().alloc(f64, 1);
                data[0] = d.result;
                try fl.append(data);
            }
            var result = try nn.predict(try ff.toOwnedSlice());
            // We calculate the error
            var err: f64 = 0.0;
            for (result, 0..) |r, i| {
                // This is the standard error via the mean of the square error
                err += (r[0] - fl.items[i][0]) * (r[0] - fl.items[i][0]);
            }
            err /= @intToFloat(f64, result.len);
            err = @sqrt(err);
            try stdout_buffered.print("Error for {d} data points: {d}\n", .{ e.inputs.len, err });
            // We change it to percentage
            err /= @intToFloat(f64, e.inputs.len);
            err *= 100.0;
            try stdout_buffered.print("Error for {d} data points: {d}%\n", .{ e.inputs.len, err });
            // We calculate the Max memory the program used
            if (@import("builtin").os.tag == .linux) {
                var file = try files.openFile("/proc/self/status", .{});
                defer file.close();
                var data_read = try file.readAll(&file_buffer);
                var data = file_buffer[0..data_read];
                var parse = std.mem.tokenize(u8, data, "\n");
                while (parse.next()) |line| {
                    var tokens = std.mem.tokenize(u8, line, ":");
                    if (tokens.next()) |token| {
                        if (std.mem.eql(u8, token, "VmPeak")) {
                            if (tokens.next()) |tok| {
                                var mem = try std.fmt.parseFloat(f64, std.mem.trim(u8, tok, "\t kB"));
                                var divisor: f32 = 1024;
                                var symbol = "MB";
                                if (mem > 1024 * 1024) {
                                    divisor = 1024 * 1024;
                                    symbol = "GB";
                                }
                                try stdout_buffered.print("Max memory used: {d} {s}\n", .{ (mem - last_mem) / divisor, symbol });
                                last_mem = mem;
                            }
                        }
                    }
                }
            }
            // If elapsed is in the realm of ms, we print it in ms, elapsed is in ns
            if (elapsed < 1000000) {
                try stdout_buffered.print("Elapsed time: {d} ms\n", .{elapsed / std.time.ns_per_ms});
            } else {
                try stdout_buffered.print("Elapsed time: {d} s\n", .{elapsed / std.time.ns_per_s});
            }
            try result_writer.print("relu+huberloss+5.5.4.2.1,{d},{d},{d},\"{d}\",{d},\"{d}\"\n", .{ e.inputs.len, 0.8, elapsed, err, last_mem, learning_rate });
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
