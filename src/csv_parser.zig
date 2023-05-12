//! A csv Parser implementation
//!

const std = @import("std");

pub fn parseCsv(comptime T: type, data: []const u8, separator: ?[]const u8, headers: bool) !CsvTokenizer(T) {
    const csv_canon_separator = sep_blk: {
        if (separator) |sep| if (sep.len > 1) {} else return error.separator_too_long;
        break :sep_blk if (separator) |sep| sep else ",";
    };
    var split = std.mem.tokenize(u8, data, "\n\r");
    if (headers) {
        _ = split.next();
    }
    return CsvTokenizer(T){
        .iterator = split,
        .separator = csv_canon_separator,
    };
}

pub fn CsvTokenizer(comptime T: type) type {
    return struct {
        iterator: std.mem.TokenIterator(u8),
        separator: []const u8,
        pub fn next(self: *CsvTokenizer(T)) ?T {
            var row = self.iterator.next();
            if (row) |r| {
                var trimmed_row = std.mem.trim(u8, r, "\n\r ");
                return parseRow(T, trimmed_row, self.separator);
            } else {
                return null;
            }
        }
    };
}

fn parseRow(comptime T: type, data: []const u8, separator: []const u8) ?T {
    var row: T = undefined;
    const fields: []const std.builtin.Type.StructField = @typeInfo(T).Struct.fields;
    var iterator_start = std.mem.split(u8, data, separator);
    inline for (fields) |f| {
        var field_data_mem = iterator_start.next();
        if ((field_data_mem == null or field_data_mem.?.len == 0) and @typeInfo(f.type) != .Optional) {
            return null;
        }
        var field_data = std.mem.trim(u8, field_data_mem.?, " \t");
        @field(row, f.name) = parseField(f.type, field_data) catch return null;
    }
    return row;
}

pub fn parseField(comptime T: type, data: []const u8) !T {
    const field_info = @typeInfo(T);
    var field: T = undefined;
    switch (field_info) {
        .Int => {
            var result: T = try std.fmt.parseInt(T, data, 10);
            field = result;
        },
        .Float => {
            var result: T = try std.fmt.parseFloat(T, data);
            field = result;
        },
        .Bool => {
            // TODO: implement
            return error.not_implemented;
        },
        .Pointer => {
            if (field_info.Pointer.child != u8) {
                return error.not_implemented;
            }
            if (field_info.Pointer.size != .Slice) {
                return error.not_implemented;
            }
            field = data;
        },
        .Optional => {
            field = try parseField(field_info.Optional.child, data);
        },
        else => {
            return error.not_implemented;
        },
    }
    return field;
}

// TESTS

pub const test_data_headerless =
    \\  10, 20, 30
    \\  40, 50, 60
    \\  70, 80, 90
    \\  100, 110, 120
    \\  130, 140, 150
;

pub const test_data_header = "a,b,c\n" ++ test_data_headerless;

const TestData = struct {
    a: u8,
    b: u8,
    c: u8,
};
const expect = std.testing.expect;
test "headerless" {
    var tokenizer = try parseCsv(TestData, test_data_headerless, null, false);
    var row_token = tokenizer.next();
    try expect(row_token != null);
    var row = row_token.?;
    try expect(row.a == 10);
    try expect(row.b == 20);
    try expect(row.c == 30);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 40);
    try expect(row.b == 50);
    try expect(row.c == 60);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 70);
    try expect(row.b == 80);
    try expect(row.c == 90);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 100);
    try expect(row.b == 110);
    try expect(row.c == 120);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 130);
    try expect(row.b == 140);
    try expect(row.c == 150);
    try expect(tokenizer.next() == null);
}

test "header" {
    var tokenizer = try parseCsv(TestData, test_data_header, null, true);
    var row_token = tokenizer.next();
    try expect(row_token != null);
    var row = row_token.?;
    try expect(row.a == 10);
    try expect(row.b == 20);
    try expect(row.c == 30);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 40);
    try expect(row.b == 50);
    try expect(row.c == 60);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 70);
    try expect(row.b == 80);
    try expect(row.c == 90);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 100);
    try expect(row.b == 110);
    try expect(row.c == 120);
    row_token = tokenizer.next();
    try expect(row_token != null);
    row = row_token.?;
    try expect(row.a == 130);
    try expect(row.b == 140);
    try expect(row.c == 150);
    try expect(tokenizer.next() == null);
}
