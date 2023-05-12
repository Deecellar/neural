const std = @import("std");
const data_gen = @import("src/data_gen.zig");
// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
const data = [_]usize{ 100, 500, 1000, 2000, 5000, 10000, 20000, 50000};
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "neural",
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const options = b.addOptions();
    options.addOption(@TypeOf(data), "data", data);
    exe.addOptions("build_options", options);
    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);
    // We see if data has been generated, and if not, we generate it.
    var data_dir_name = b.pathFromRoot("data");
    if (std.fs.openDirAbsolute(data_dir_name, .{})) |_| {
        std.log.info("Data directory exists, skipping generation.\n", .{});
    } else |_| {
        std.log.info("Data directory does not exist, generating.\n", .{});
        std.fs.makeDirAbsolute(data_dir_name) catch |err| {
            std.log.warn("Error creating data directory: {}\n", .{err});
            @panic("Failed to create data directory.");
        };
        inline for (data) |d| {
            data_gen.generateData(d, "data") catch |err| {
                std.log.warn("Error generating data: {}\n", .{err});
                @panic("Failed to generate data.");
            };
        }
    }

    const data_dir = b.addInstallDirectory(.{
        .source_dir = data_dir_name,
        .install_dir = .bin,
        .install_subdir = "data",
    });
    b.getInstallStep().dependOn(&data_dir.step);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
