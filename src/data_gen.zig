//! This file generates a CSV with sintetic data for the learning process
//! The data generated is the result for f(x[d]) = 418.9829 * d - SUM(d * sin(sqrt(|d|)))
//! where d is is the d-th dimension of the input vector
//! Where are using 5 dimensions for the input vector
//! The range of the input vector is [-500, 500]
//!

const std = @import("std");

pub fn generateData(comptime rows: usize, comptime path_prefix: []const u8) !void {
    var random = std.rand.DefaultPrng.init(@bitCast(u64, std.time.microTimestamp()));
    var file_name = std.fmt.comptimePrint("data_r{d}.csv", .{rows});
    var file = try std.fs.cwd().createFile(path_prefix ++ "/" ++ file_name, .{});
    defer file.close();
    try file.writeAll("d1,d2,d3,d4,d5,result\n");
    var random_interface = random.random();
    for (0..rows) |_| {
        var d1 = random_interface.float(f64) * 1000.0 - 500.0;
        var d2 = random_interface.float(f64) * 1000.0 - 500.0;
        var d3 = random_interface.float(f64) * 1000.0 - 500.0;
        var d4 = random_interface.float(f64) * 1000.0 - 500.0;
        var d5 = random_interface.float(f64) * 1000.0 - 500.0;
        var result: f64 = 418.9829 * 5.0;
        var sum: f64 = 0.0;
        inline for (0..5) |i| {
            var d = switch (i) {
                0 => d1,
                1 => d2,
                2 => d3,
                3 => d4,
                4 => d5,
                else => 0.0,
            };
            sum += d * @sin(@sqrt(@fabs(d)));
        }
        result -= sum;
        try file.writer().print("{d},{d},{d},{d},{d},{d}\n", .{ d1, d2, d3, d4, d5, result });
    }
}
